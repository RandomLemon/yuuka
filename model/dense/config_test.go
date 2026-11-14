package dense

import "testing"

func validTestConfig() Config {
	return Config{
		Architecture: Llama,
		BlockCount:   2,
		Context:      256,
		HiddenSize:   16,
		FFNSize:      32,
		HeadCount:    2,
		KVHeadCount:  1,
		HeadDim:      8,
		ValueDim:     8,
		RoPEDim:      8,
		RoPEFreqBase: 10000,
		RMSNormEps:   1e-6,
		VocabSize:    259,
	}
}

func TestConfigValidate(t *testing.T) {
	cfg := validTestConfig()
	if err := cfg.validate(); err != nil {
		t.Fatalf("valid config: %v", err)
	}
	cases := []struct {
		name   string
		mutate func(*Config)
	}{
		{"head count not multiple", func(c *Config) { c.HeadCount = 3; c.KVHeadCount = 2 }},
		{"key and value dims differ", func(c *Config) { c.ValueDim = 4 }},
		{"odd rope dimension", func(c *Config) { c.RoPEDim = 7 }},
		{"rope dimension exceeds head", func(c *Config) { c.RoPEDim = 10 }},
		{"zero context", func(c *Config) { c.Context = 0 }},
		{"zero vocabulary", func(c *Config) { c.VocabSize = 0 }},
		{"moe used count zero", func(c *Config) { c.ExpertCount = 4; c.ExpertUsedCount = 0; c.ExpertFFNSize = 8 }},
		{"moe used count exceeds experts", func(c *Config) {
			c.ExpertCount = 4
			c.ExpertUsedCount = 5
			c.ExpertFFNSize = 8
		}},
		{"moe zero expert ffn", func(c *Config) { c.ExpertCount = 4; c.ExpertUsedCount = 2; c.ExpertFFNSize = 0 }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := validTestConfig()
			tc.mutate(&cfg)
			if err := cfg.validate(); err == nil {
				t.Fatal("invalid config was accepted")
			}
		})
	}
}
