#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum WorkflowBackground3D {
    Solid { color: [f32; 3] },
    VerticalGradient { top: [f32; 3], bottom: [f32; 3] },
}

impl WorkflowBackground3D {
    pub fn bottom_color(&self) -> [f32; 3] {
        match *self {
            Self::Solid { color } => color,
            Self::VerticalGradient { bottom, .. } => bottom,
        }
    }

    pub fn top_color(&self) -> [f32; 3] {
        match *self {
            Self::Solid { color } => color,
            Self::VerticalGradient { top, .. } => top,
        }
    }
}

impl Default for WorkflowBackground3D {
    fn default() -> Self {
        Self::VerticalGradient {
            top: [0.10, 0.12, 0.16],
            bottom: [0.02, 0.03, 0.05],
        }
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WorkflowRender3D {
    pub lighting_preset: SceneLightingPreset,
    pub background: WorkflowBackground3D,
    pub fog_enabled: bool,
    pub fog_color: [f32; 3],
    pub fog_start_fraction: f32,
    pub fog_end_fraction: f32,
    pub vignette_strength: f32,
    pub exposure: f32,
    pub contrast: f32,
}

impl WorkflowRender3D {
    pub fn scene_lighting(&self) -> SceneLightingParams {
        SceneLightingParams {
            preset: self.lighting_preset,
        }
    }

    pub fn sanitized(mut self) -> Self {
        self.fog_start_fraction = self.fog_start_fraction.clamp(0.0, 0.95);
        self.fog_end_fraction = self
            .fog_end_fraction
            .clamp((self.fog_start_fraction + 0.05).min(1.0), 1.0);
        self.vignette_strength = self.vignette_strength.clamp(0.0, 0.5);
        self.exposure = self.exposure.clamp(0.5, 1.5);
        self.contrast = self.contrast.clamp(0.75, 1.5);
        self
    }
}

impl Default for WorkflowRender3D {
    fn default() -> Self {
        Self {
            lighting_preset: SceneLightingPreset::Soft,
            background: WorkflowBackground3D::default(),
            fog_enabled: false,
            fog_color: [0.02, 0.03, 0.05],
            fog_start_fraction: 0.55,
            fog_end_fraction: 1.00,
            vignette_strength: 0.12,
            exposure: 1.0,
            contrast: 1.0,
        }
    }
}
