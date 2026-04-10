#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SceneLightingPreset {
    Flat,
    Soft,
    Studio,
}

#[allow(dead_code)]
impl SceneLightingPreset {
    pub fn label(self) -> &'static str {
        match self {
            Self::Flat => "Flat",
            Self::Soft => "Soft",
            Self::Studio => "Studio",
        }
    }

    pub const ALL: [Self; 3] = [Self::Flat, Self::Soft, Self::Studio];
}

#[derive(Clone, Copy)]
pub struct SceneLightingParams {
    pub preset: SceneLightingPreset,
}

impl SceneLightingParams {
    pub fn ambient_strength(self) -> f32 {
        match self.preset {
            SceneLightingPreset::Flat => 1.0,
            SceneLightingPreset::Soft => 0.62,
            SceneLightingPreset::Studio => 0.50,
        }
    }

    pub fn key_strength(self) -> f32 {
        match self.preset {
            SceneLightingPreset::Flat => 0.0,
            SceneLightingPreset::Soft => 0.34,
            SceneLightingPreset::Studio => 0.52,
        }
    }

    pub fn fill_strength(self) -> f32 {
        match self.preset {
            SceneLightingPreset::Flat => 0.0,
            SceneLightingPreset::Soft => 0.24,
            SceneLightingPreset::Studio => 0.30,
        }
    }

    pub fn headlight_mix(self) -> f32 {
        match self.preset {
            SceneLightingPreset::Flat => 0.0,
            SceneLightingPreset::Soft => 0.28,
            SceneLightingPreset::Studio => 0.18,
        }
    }

    pub fn specular_strength(self) -> f32 {
        match self.preset {
            SceneLightingPreset::Flat => 0.0,
            SceneLightingPreset::Soft => 0.14,
            SceneLightingPreset::Studio => 0.26,
        }
    }
}

impl Default for SceneLightingParams {
    fn default() -> Self {
        Self {
            preset: SceneLightingPreset::Soft,
        }
    }
}
