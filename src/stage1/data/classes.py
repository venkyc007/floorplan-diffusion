"""The 6 classes (channels) of our floor plan images — defined ONCE here.

Every other file imports these names instead of typing numbers like "2",
so the channel order can never get mixed up between files.
"""

# Channel order of every (6, 128, 128) array in this project.
CLASSES = ["wall", "door", "window", "room", "outdoor", "background"]
WALL, DOOR, WINDOW, ROOM, OUTDOOR, BACKGROUND = range(6)
NUM_CLASSES = len(CLASSES)

# Which ResPlan fields go into which of our classes.
# Fields NOT listed ('inner' = indoor outline, 'neighbor' = adjacent units)
# are helper shapes and become background.
RESPLAN_FIELDS = {
    WALL: ["wall"],
    DOOR: ["door", "front_door"],      # the front door counts as a door
    WINDOW: ["window"],
    # Enclosed indoor rooms only.
    ROOM: ["living", "bedroom", "bathroom", "kitchen", "storage", "stair"],
    # Open-air parts: 'land' is the yard/setback/light-well area around and
    # inside the building footprint. (ResPlan has no 'terrace' class.)
    OUTDOOR: ["balcony", "garden", "land", "parking", "pool"],
}

# Fields that decide how a plan is scaled into the 128x128 frame: the
# building's structure only (walls + doors + windows). Everything else
# (balconies, outside stairs, garden, land, parking) is cropped where it
# extends beyond the frame, so it can never shrink the building.
EXTENT_FIELDS = ["wall", "door", "front_door", "window"]

# Order in which classes are painted. Later ones are painted on top, so if two
# shapes cover the same pixel, the later class wins: outdoor is the bottom
# layer, and doors/windows are drawn on top of walls.
PAINT_ORDER = [OUTDOOR, ROOM, WALL, WINDOW, DOOR]

# Colours (R, G, B) used when drawing plans as pictures.
COLORS = {
    WALL: (30, 30, 30),          # near-black
    DOOR: (220, 50, 47),         # red
    WINDOW: (38, 139, 210),      # blue
    ROOM: (238, 220, 170),       # light sand
    OUTDOOR: (166, 219, 160),    # light green
    BACKGROUND: (255, 255, 255), # white
}
