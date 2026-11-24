package qwen35

import "math"

// gdnChunkSize is the token chunk of the chunkwise gated delta rule. It matches
// the device kernel, which keeps the triangular solve at 32 rows so it fits one
// warp-wide inversion.
const gdnChunkSize = 32

// gdnScratch holds the working set of gdnChunkedScan: two 32x32 matrices and
// four 32 x stateSize blocks, reused across chunks and heads.
type gdnScratch struct {
	minv  []float32 // (chunk x chunk) inverse of the unit lower triangular system
	bmat  []float32 // (chunk x chunk) unit lower triangular system, then Z
	kproj []float32 // (chunk x state) K S
	qproj []float32 // (chunk x state) Q S
	work  []float32 // (chunk x state) scratch row block
	ref   []float32 // (chunk x state) scratch row block
	logd  []float32 // (chunk) cumulative log decay
}

func newGDNScratch(stateSize int) *gdnScratch {
	mat := gdnChunkSize * gdnChunkSize
	vec := gdnChunkSize * stateSize
	return &gdnScratch{
		minv:  make([]float32, mat),
		bmat:  make([]float32, mat),
		kproj: make([]float32, vec),
		qproj: make([]float32, vec),
		work:  make([]float32, vec),
		ref:   make([]float32, vec),
		logd:  make([]float32, gdnChunkSize),
	}
}

// gdnNormGate applies the gated RMSNorm to the raw per-head dot products of one
// chunk. Both scans produce the same output layout so this final step is
// shared: out is (rows x rank x state), z is the gating projection.
func gdnNormGate(out, z, norm []float32, rows, rank, stateSize int, eps float32) {
	for r := 0; r < rows; r++ {
		for h := 0; h < rank; h++ {
			row := out[(r*rank+h)*stateSize : (r*rank+h+1)*stateSize]
			rmsNormInPlace(row, norm, eps)
			gate := z[(r*rank+h)*stateSize : (r*rank+h+1)*stateSize]
			for d := range row {
				row[d] *= silu(gate[d])
			}
		}
	}
}

// gdnPerTokenScan applies the gated delta-net recurrence one token at a time.
// conv holds the L2-normalized q and k followed by the raw v of each row; alpha
// is the log decay and beta the sigmoid gate of each (row, head). The output is
// the raw per-head dot product before RMSNorm and gating.
func gdnPerTokenScan(state, sk, d, conv, alpha, beta, out []float32, rows, groups,
	rank, stateSize, convDim int) {
	keyDim := groups * stateSize
	for r := 0; r < rows; r++ {
		row := conv[r*convDim : (r+1)*convDim]
		for h := 0; h < rank; h++ {
			kh := h % groups
			q := row[kh*stateSize : (kh+1)*stateSize]
			k := row[keyDim+kh*stateSize : keyDim+(kh+1)*stateSize]
			v := row[2*keyDim+h*stateSize : 2*keyDim+(h+1)*stateSize]
			matrix := state[h*stateSize*stateSize : (h+1)*stateSize*stateSize]
			decay := float32(math.Exp(float64(alpha[r*rank+h])))
			for i := range matrix {
				matrix[i] *= decay
			}
			for j := range sk {
				sk[j] = 0
			}
			for i := 0; i < stateSize; i++ {
				ki := k[i]
				mrow := matrix[i*stateSize : (i+1)*stateSize]
				for j := 0; j < stateSize; j++ {
					sk[j] += mrow[j] * ki
				}
			}
			b := beta[r*rank+h]
			for j := 0; j < stateSize; j++ {
				d[j] = (v[j] - sk[j]) * b
			}
			for i := 0; i < stateSize; i++ {
				ki := k[i]
				mrow := matrix[i*stateSize : (i+1)*stateSize]
				for j := 0; j < stateSize; j++ {
					mrow[j] += ki * d[j]
				}
			}
			head := out[(r*rank+h)*stateSize : (r*rank+h+1)*stateSize]
			for j := range head {
				head[j] = 0
			}
			for i := 0; i < stateSize; i++ {
				qi := q[i]
				mrow := matrix[i*stateSize : (i+1)*stateSize]
				for j := 0; j < stateSize; j++ {
					head[j] += mrow[j] * qi
				}
			}
		}
	}
}

// gdnChunkedScan applies the same recurrence in the chunkwise WY form. Within a
// chunk of n tokens the state is expressed relative to the chunk-start state,
// so the rank-one updates become the triangular system
//
//	M = I + B,   B[r][s] = beta_r (k_r . k_s)   for s < r,
//
// and dense products. With z = (Q K^T on the lower triangle) M^-1, the state
// after a chunk and the output rows are
//
//	S_C = exp(g_c) S_0 + K^T (M^-1 diag(beta) (exp(g_c - g) V - exp(g_c) K S_0))
//	o_r = exp(g_r) (q_r S_0 + z_r (diag(beta exp(-g)) V - M^-1 diag(beta) K S_0))
//
// where g is the cumulative log decay from the chunk start and g_c its last
// entry. The decay of a row is a scalar for the whole state matrix, which is
// why it factors out of the solve. See docs/kernels.md section 5.6.
func gdnChunkedScan(state []float32, conv, alpha, beta, out []float32, rows,
	groups, rank, stateSize, convDim int, sc *gdnScratch) {
	for c0 := 0; c0 < rows; c0 += gdnChunkSize {
		n := min(gdnChunkSize, rows-c0)
		for h := 0; h < rank; h++ {
			gdnChunkHead(state, conv, alpha, beta, out, sc, c0, n, h, groups,
				rank, stateSize, convDim)
		}
	}
}

func gdnChunkHead(state, conv, alpha, beta, out []float32, sc *gdnScratch,
	c0, n, h, groups, rank, stateSize, convDim int) {
	stride := gdnChunkSize * stateSize
	kh := h % groups
	keyDim := groups * stateSize
	qOffset := kh * stateSize
	kOffset := keyDim + kh*stateSize
	vOffset := 2*keyDim + h*stateSize
	query := func(r int) []float32 {
		row := conv[(c0+r)*convDim : (c0+r+1)*convDim]
		return row[qOffset : qOffset+stateSize]
	}
	key := func(r int) []float32 {
		row := conv[(c0+r)*convDim : (c0+r+1)*convDim]
		return row[kOffset : kOffset+stateSize]
	}
	value := func(r int) []float32 {
		row := conv[(c0+r)*convDim : (c0+r+1)*convDim]
		return row[vOffset : vOffset+stateSize]
	}

	// Cumulative log decay from the chunk start.
	var acc float32
	for r := 0; r < n; r++ {
		acc += alpha[(c0+r)*rank+h]
		sc.logd[r] = acc
	}
	gc := acc
	decay := float32(math.Exp(float64(gc)))

	// B[r][s] = beta_r (k_r . k_s) for s < r.
	lower := sc.bmat[:gdnChunkSize*gdnChunkSize]
	for i := range lower {
		lower[i] = 0
	}
	for r := 0; r < n; r++ {
		kr := key(r)
		br := beta[(c0+r)*rank+h]
		for s := 0; s < r; s++ {
			ks := key(s)
			var sum float32
			for i := 0; i < stateSize; i++ {
				sum += kr[i] * ks[i]
			}
			lower[r*gdnChunkSize+s] = br * sum
		}
	}
	minv := sc.minv[:gdnChunkSize*gdnChunkSize]
	inverseLower(lower, minv, n)

	// Z = (Q K^T on the lower triangle) M^-1, over the bmat block.
	zm := sc.bmat[:gdnChunkSize*gdnChunkSize]
	prow := sc.qproj[:gdnChunkSize]
	for r := 0; r < n; r++ {
		qr := query(r)
		for s := 0; s <= r; s++ {
			ks := key(s)
			var sum float32
			for i := 0; i < stateSize; i++ {
				sum += qr[i] * ks[i]
			}
			prow[s] = sum
		}
		for s := 0; s <= r; s++ {
			var sum float32
			for u := s; u <= r; u++ {
				sum += prow[u] * minv[u*gdnChunkSize+s]
			}
			rowOf(zm, r)[s] = sum
		}
	}

	// Fold diag(beta) into the inverse columns: mb[r][s] = minv[r][s] * beta_s.
	for r := 0; r < n; r++ {
		for s := 0; s < n; s++ {
			minv[r*gdnChunkSize+s] *= beta[(c0+s)*rank+h]
		}
	}

	kp := sc.kproj[:stride]
	qp := sc.qproj[:stride]
	matrix := state[h*stateSize*stateSize : (h+1)*stateSize*stateSize]
	for r := 0; r < n; r++ {
		kr := key(r)
		qr := query(r)
		krow := kp[r*stateSize : (r+1)*stateSize]
		qrow := qp[r*stateSize : (r+1)*stateSize]
		for j := 0; j < stateSize; j++ {
			krow[j] = 0
			qrow[j] = 0
		}
		for i := 0; i < stateSize; i++ {
			ki := kr[i]
			qi := qr[i]
			srow := matrix[i*stateSize : (i+1)*stateSize]
			for j := 0; j < stateSize; j++ {
				krow[j] += ki * srow[j]
				qrow[j] += qi * srow[j]
			}
		}
	}

	// xi = exp(g_c - g_r) v_r - exp(g_c) (K S)_r, then values = mb xi.
	tmp := make([]float32, stateSize)
	work := sc.work[:stride]
	ref := sc.ref[:stride]
	for r := 0; r < n; r++ {
		scale := float32(math.Exp(float64(gc - sc.logd[r])))
		vr := value(r)
		krow := kp[r*stateSize : (r+1)*stateSize]
		wrow := work[r*stateSize : (r+1)*stateSize]
		for j := 0; j < stateSize; j++ {
			wrow[j] = scale*vr[j] - decay*krow[j]
		}
	}
	applyLower(ref, work, minv, tmp, n, stateSize)

	// S = exp(g_c) S_0 + sum_r k_r values_r^T.
	for i := range matrix {
		matrix[i] *= decay
	}
	for r := 0; r < n; r++ {
		kr := key(r)
		vr := ref[r*stateSize : (r+1)*stateSize]
		for i := 0; i < stateSize; i++ {
			ki := kr[i]
			srow := matrix[i*stateSize : (i+1)*stateSize]
			for j := 0; j < stateSize; j++ {
				srow[j] += ki * vr[j]
			}
		}
	}

	// inner = diag(beta) (diag(exp(-g)) V - K S_0), then zo = Z inner.
	for r := 0; r < n; r++ {
		vr := value(r)
		br := beta[(c0+r)*rank+h]
		scale := float32(math.Exp(float64(-sc.logd[r])))
		wrow := work[r*stateSize : (r+1)*stateSize]
		krow := kp[r*stateSize : (r+1)*stateSize]
		for j := 0; j < stateSize; j++ {
			wrow[j] = br * (scale*vr[j] - krow[j])
		}
	}
	for r := 0; r < n; r++ {
		zrow := zm[r*gdnChunkSize : r*gdnChunkSize+n]
		outrow := ref[r*stateSize : (r+1)*stateSize]
		for j := 0; j < stateSize; j++ {
			outrow[j] = 0
		}
		for s := 0; s <= r; s++ {
			zs := zrow[s]
			wrow := work[s*stateSize : (s+1)*stateSize]
			for j := 0; j < stateSize; j++ {
				outrow[j] += zs * wrow[j]
			}
		}
	}
	for r := 0; r < n; r++ {
		scale := float32(math.Exp(float64(sc.logd[r])))
		qrow := qp[r*stateSize : (r+1)*stateSize]
		orow := ref[r*stateSize : (r+1)*stateSize]
		head := out[((c0+r)*rank+h)*stateSize : ((c0+r)*rank+h+1)*stateSize]
		for j := 0; j < stateSize; j++ {
			head[j] = scale * (qrow[j] + orow[j])
		}
	}
}

// rowOf returns the row r of a chunk-major n x n block.
func rowOf(m []float32, r int) []float32 { return m[r*gdnChunkSize : r*gdnChunkSize+gdnChunkSize] }

// inverseLower writes (I + lower)^-1 into minv for an n x n unit lower
// triangular system with the diagonal of lower ignored.
func inverseLower(lower, minv []float32, n int) {
	for r := 0; r < n; r++ {
		for s := 0; s < n; s++ {
			minv[r*gdnChunkSize+s] = 0
		}
		minv[r*gdnChunkSize+r] = 1
	}
	// Column-wise forward substitution: row r of the inverse is the identity
	// row r minus the strict-lower rows already resolved.
	for r := 1; r < n; r++ {
		for u := 0; u < r; u++ {
			lu := lower[r*gdnChunkSize+u]
			if lu == 0 {
				continue
			}
			row := minv[r*gdnChunkSize : r*gdnChunkSize+n]
			up := minv[u*gdnChunkSize : u*gdnChunkSize+n]
			for s := 0; s <= u; s++ {
				row[s] -= lu * up[s]
			}
		}
	}
}

// applyLower writes dst = mb src for the first n rows of two row blocks, where
// mb is the n x n row-major lower triangular matrix built by gdnChunkHead.
func applyLower(dst, src, mb, tmp []float32, n, stateSize int) {
	for r := 0; r < n; r++ {
		out := tmp[:stateSize]
		for j := range out {
			out[j] = 0
		}
		for s := 0; s <= r; s++ {
			coef := mb[r*gdnChunkSize+s]
			if coef == 0 {
				continue
			}
			srow := src[s*stateSize : (s+1)*stateSize]
			for j := 0; j < stateSize; j++ {
				out[j] += coef * srow[j]
			}
		}
		copy(dst[r*stateSize:(r+1)*stateSize], out)
	}
}
