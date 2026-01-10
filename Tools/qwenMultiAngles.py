class qwenMultiAngles:
    _AZIMUTH_OPTIONS = [
        ("front view", "front view（正面视角）"),
        ("front-right quarter view", "front-right quarter view（前右四分之三视角）"),
        ("right side view", "right side view（右侧视角）"),
        ("back-right quarter view", "back-right quarter view（后右四分之三视角）"),
        ("back view", "back view（背面视角）"),
        ("back-left quarter view", "back-left quarter view（后左四分之三视角）"),
        ("left side view", "left side view（左侧视角）"),
        ("front-left quarter view", "front-left quarter view（前左四分之三视角）"),
    ]

    _ELEVATION_OPTIONS = [
        ("low-angle shot", "low-angle shot（低角度镜头）"),
        ("eye-level shot", "eye-level shot（平视镜头）"),
        ("elevated shot", "elevated shot（稍高角度镜头）"),
        ("high-angle shot", "high-angle shot（高角度镜头）"),
    ]

    _DISTANCE_OPTIONS = [
        ("close-up", "close-up（特写）"),
        ("medium shot", "medium shot（中景）"),
        ("wide shot", "wide shot（广角）"),
    ]

    _DISPLAY_TO_VALUE = {
        display: value
        for value, display in (_AZIMUTH_OPTIONS + _ELEVATION_OPTIONS + _DISTANCE_OPTIONS)
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prefix": ("STRING", {"default": "<sks>", "multiline": False}),
                "azimuth": (
                    [
                        v for _, v in cls._AZIMUTH_OPTIONS
                    ],
                ),
                "elevation": (
                    [
                        v for _, v in cls._ELEVATION_OPTIONS
                    ],
                ),
                "distance": (
                    [
                        v for _, v in cls._DISTANCE_OPTIONS
                    ],
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "build"
    CATEGORY = "Tools"

    def build(self, prefix, azimuth, elevation, distance):
        prefix = prefix.strip() if isinstance(prefix, str) else "<sks>"
        if not prefix:
            prefix = "<sks>"
        azimuth = self._DISPLAY_TO_VALUE.get(azimuth, azimuth)
        elevation = self._DISPLAY_TO_VALUE.get(elevation, elevation)
        distance = self._DISPLAY_TO_VALUE.get(distance, distance)
        prompt = f"{prefix} {azimuth} {elevation} {distance}"
        return (prompt,)


NODE_CLASS_MAPPINGS = {
    "qwenMultiAngles": qwenMultiAngles,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "qwenMultiAngles": "qwenMultiAngles☀",
}
