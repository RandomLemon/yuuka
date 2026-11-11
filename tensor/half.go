package tensor

import "math"

// Float16ToFloat32 converts an IEEE 754 binary16 value to float32.
func Float16ToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x03ff
	var bits uint32
	switch exp {
	case 0:
		if mant == 0 {
			bits = sign
		} else {
			// Subnormal: normalize the mantissa into a float32 normal.
			e := uint32(127 - 15 + 1)
			for mant&0x0400 == 0 {
				mant <<= 1
				e--
			}
			mant &= 0x03ff
			bits = sign | e<<23 | mant<<13
		}
	case 0x1f:
		bits = sign | 0x7f800000 | mant<<13
	default:
		bits = sign | (exp+112)<<23 | mant<<13
	}
	return math.Float32frombits(bits)
}

// Float32ToFloat16 rounds a float32 to binary16. It is used by tests and by
// reference kernels that need to reproduce storage rounding.
func Float32ToFloat16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16((b >> 16) & 0x8000)
	exp := int32((b>>23)&0xff) - 127
	mant := b & 0x7fffff
	if exp == 128 {
		if mant != 0 {
			return sign | 0x7e00
		}
		return sign | 0x7c00
	}
	if exp > 15 {
		return sign | 0x7c00
	}
	if exp < -24 {
		return sign
	}
	if exp < -14 {
		// Subnormal half.
		shift := uint32(-exp - 14)
		full := mant | 0x800000
		rounded := full + (1 << (shift + 12))
		return sign | uint16(rounded>>(shift+13))
	}
	rounded := mant + 0x1000
	if rounded&0x800000 != 0 {
		exp++
		rounded = 0
		if exp > 15 {
			return sign | 0x7c00
		}
	}
	return sign | uint16(exp+15)<<10 | uint16(rounded>>13)
}
