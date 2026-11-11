package gguf

import (
	"fmt"
	"math"
)

// MetadataType is the GGUF metadata value type.
type MetadataType uint32

const (
	MetaUint8   MetadataType = 0
	MetaInt8    MetadataType = 1
	MetaUint16  MetadataType = 2
	MetaInt16   MetadataType = 3
	MetaUint32  MetadataType = 4
	MetaInt32   MetadataType = 5
	MetaFloat32 MetadataType = 6
	MetaBool    MetadataType = 7
	MetaString  MetadataType = 8
	MetaArray   MetadataType = 9
	MetaUint64  MetadataType = 10
	MetaInt64   MetadataType = 11
	MetaFloat64 MetadataType = 12
)

// String returns the GGUF type name.
func (t MetadataType) String() string {
	switch t {
	case MetaUint8:
		return "uint8"
	case MetaInt8:
		return "int8"
	case MetaUint16:
		return "uint16"
	case MetaInt16:
		return "int16"
	case MetaUint32:
		return "uint32"
	case MetaInt32:
		return "int32"
	case MetaFloat32:
		return "float32"
	case MetaBool:
		return "bool"
	case MetaString:
		return "string"
	case MetaArray:
		return "array"
	case MetaUint64:
		return "uint64"
	case MetaInt64:
		return "int64"
	case MetaFloat64:
		return "float64"
	default:
		return fmt.Sprintf("metadata_type(%d)", uint32(t))
	}
}

// Value is one decoded metadata value. Only the field selected by Type is
// valid. Array carries the element type in ElemType.
type Value struct {
	Type     MetadataType
	ElemType MetadataType
	Bool     bool
	Uint     uint64
	Int      int64
	Float    float64
	String   string
	Array    []Value
}

// AsUint converts integer metadata to uint64.
func (v Value) AsUint() (uint64, error) {
	switch v.Type {
	case MetaUint8, MetaUint16, MetaUint32, MetaUint64:
		return v.Uint, nil
	case MetaInt8, MetaInt16, MetaInt32, MetaInt64:
		if v.Int < 0 {
			return 0, fmt.Errorf("metadata value %d is negative", v.Int)
		}
		return uint64(v.Int), nil
	default:
		return 0, fmt.Errorf("metadata type %s is not an integer", v.Type)
	}
}

// AsInt converts integer metadata to int64.
func (v Value) AsInt() (int64, error) {
	switch v.Type {
	case MetaInt8, MetaInt16, MetaInt32, MetaInt64:
		return v.Int, nil
	case MetaUint8, MetaUint16, MetaUint32:
		return int64(v.Uint), nil
	case MetaUint64:
		if v.Uint > math.MaxInt64 {
			return 0, fmt.Errorf("metadata value %d overflows int64", v.Uint)
		}
		return int64(v.Uint), nil
	default:
		return 0, fmt.Errorf("metadata type %s is not an integer", v.Type)
	}
}

// AsFloat converts float metadata to float64.
func (v Value) AsFloat() (float64, error) {
	switch v.Type {
	case MetaFloat32, MetaFloat64:
		return v.Float, nil
	default:
		return 0, fmt.Errorf("metadata type %s is not a float", v.Type)
	}
}

// AsString returns string metadata.
func (v Value) AsString() (string, error) {
	if v.Type != MetaString {
		return "", fmt.Errorf("metadata type %s is not a string", v.Type)
	}
	return v.String, nil
}

// AsBool returns bool metadata.
func (v Value) AsBool() (bool, error) {
	if v.Type != MetaBool {
		return false, fmt.Errorf("metadata type %s is not a bool", v.Type)
	}
	return v.Bool, nil
}

// AsStrings returns an array of strings.
func (v Value) AsStrings() ([]string, error) {
	if v.Type != MetaArray {
		return nil, fmt.Errorf("metadata type %s is not an array", v.Type)
	}
	if v.ElemType != MetaString {
		return nil, fmt.Errorf("array element type is %s, not string", v.ElemType)
	}
	out := make([]string, len(v.Array))
	for i, item := range v.Array {
		if item.Type != MetaString {
			return nil, fmt.Errorf("array element %d has type %s", i, item.Type)
		}
		out[i] = item.String
	}
	return out, nil
}

// AsUints converts an integer array to uint64 values.
func (v Value) AsUints() ([]uint64, error) {
	if v.Type != MetaArray {
		return nil, fmt.Errorf("metadata type %s is not an array", v.Type)
	}
	out := make([]uint64, len(v.Array))
	for i := range v.Array {
		value, err := v.Array[i].AsUint()
		if err != nil {
			return nil, fmt.Errorf("array element %d: %w", i, err)
		}
		out[i] = value
	}
	return out, nil
}

// AsInts converts an integer array to int64 values.
func (v Value) AsInts() ([]int64, error) {
	if v.Type != MetaArray {
		return nil, fmt.Errorf("metadata type %s is not an array", v.Type)
	}
	out := make([]int64, len(v.Array))
	for i := range v.Array {
		value, err := v.Array[i].AsInt()
		if err != nil {
			return nil, fmt.Errorf("array element %d: %w", i, err)
		}
		out[i] = value
	}
	return out, nil
}

// AsFloats converts a float array to float64 values.
func (v Value) AsFloats() ([]float64, error) {
	if v.Type != MetaArray {
		return nil, fmt.Errorf("metadata type %s is not an array", v.Type)
	}
	out := make([]float64, len(v.Array))
	for i := range v.Array {
		item := v.Array[i]
		if item.Type != MetaFloat32 && item.Type != MetaFloat64 {
			return nil, fmt.Errorf("array element %d has type %s", i, item.Type)
		}
		out[i] = item.Float
	}
	return out, nil
}
