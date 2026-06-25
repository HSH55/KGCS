# KGCS Configuration
# ---------------------------------------------------------------------------
# Paths & Global Settings

# --- SAM Model ---
SAM_MODEL_TYPE = "vit_h"          # vit_h / vit_l / vit_b
SAM_CHECKPOINT = "sam_vit_h.pth"  # SAM checkpoint filename (place in project root)

# --- CLIP Model ---
CLIP_MODEL_NAME = "ViT-L/14@336px"

# --- Output Directories ---
OUTPUT_BASE = "output"                   # where final results go
TEMP_DIRS = [
    "seg_img", "seg_img_back", "images_masks", "images",
    "images_orgin", "data", "data_append", "label",
    "wrongobjectMASK", "wrongobjectIMG", "adjust", "false_masks",
]

# --- OPM Parameters ---
OPM = {
    "crop_n_layers": 1,
    "pred_iou_thresh": 0.86,
    "stability_score_thresh": 0.92,
    "min_mask_region_area": 100,
    "min_contour_area": 100,
    "iou_threshold": 0.5,
}

# --- ISM Parameters (Gradient Screening) ---
ISM = {
    "sigma_factor_clear": 0.5,
    "sigma_factor_ambiguous": 0.8,
    "retention_ratio_clear": 0.8,
    "retention_ratio_ambiguous": 0.3,
}

# --- SDM: Object Descriptions ---
TARGET_DESCRIPTIONS = {
    "airplane": "A sleek, aerodynamic vehicle with large, flat wings extending from the fuselage, a pointed nose, and a tail fin, often seen in the sky or on runways.",
    "airport": "A vast, rectangular area featuring long, straight runways, intersecting taxiways, and multiple terminal buildings with symmetrical layouts.",
    "baseballfield": "A diamond-shaped grassy area with four bases forming a square, a circular pitcher's mound at the center, and a curved outfield perimeter.",
    "basketballcourt": "A flat, rectangular court with straight boundary lines, a central circle, and two rectangular backboards at each end, featuring circular hoops.",
    "bridge": "A linear structure, often arched or supported by vertical pillars, spanning across a gap or waterway, featuring a flat deck with parallel railings.",
    "chimney": "A tall, cylindrical or rectangular shaft, usually vertical, protruding from the roof of a building, tapering slightly towards the top.",
    "dam": "A massive, curved or straight barrier with a broad base, narrowing towards the top, constructed across a river with a smooth, sloping face.",
    "Expressway-Service-area": "A symmetrical, open area along a highway, featuring a series of rectangular buildings, parking lots, and roadways with clear, linear patterns.",
    "Expressway-toll-station": "A linear array of booths or arches spanning across the roadway, each with a barrier arm and lane markings.",
    "golffield": "An expansive, undulating landscape with several small, circular greens, tee boxes, and sand traps, connected by wide, meandering fairways.",
    "groundtrackfield": "A large oval or circular track with parallel lanes encircling a central rectangular field, used for running and athletic events.",
    "harbor": "An enclosed water area with straight or curved piers extending into the water, forming a grid-like or radial pattern for docking ships.",
    "overpass": "A linear bridge structure elevated above another road or railroad, supported by vertical pillars, featuring a flat roadway with guardrails.",
    "ship": "A long, streamlined vessel with a pointed bow and a broader stern, featuring a flat or slightly curved deck, often with vertical masts or funnels.",
    "stadium": "A circular or oval structure with tiered, concentric seating surrounding a central field or arena, characterized by a large, symmetrical shape.",
    "storagetank": "A large, cylindrical container with a flat or domed top and smooth, vertical walls, typically set on a circular base.",
    "tenniscourt": "A rectangular playing area divided by a central net, with parallel boundary lines marking the court for singles and doubles play.",
    "trainstation": "A linear structure with platforms on either side of parallel train tracks, featuring a canopy or roof overhead and rectangular buildings.",
    "vehicle": "A compact, rectangular body with four wheels, a distinct front with headlights and grille, and a slightly curved roof.",
    "windmill": "A tall, slender tower with long, narrow blades extending from the top, forming a radial pattern, often tapering outward.",
}

DISTRACTOR_DESCRIPTIONS = {
    "building": "A rectangular or L-shaped structure with a flat or sloped roof, often surrounded by roads, parking lots, or vegetation.",
    "road": "A long, linear strip of gray or dark surface, typically bordered by vegetation, with occasional vehicles and lane markings.",
    "parking_lot": "A rectangular area with rows of small, uniformly shaped rectangular objects (vehicles) and interconnecting drive lanes.",
    "woodland": "An irregular area with rough, textured green or dark canopy, lacking clear man-made structures or straight edges.",
    "water_body": "A large, smooth, dark or blue area with no texture, often with irregular shorelines and varying reflectance.",
    "shadow": "A dark, elongated area with soft edges, typically cast by tall structures or terrain, appearing darker than surroundings.",
    "bare_land": "A large, uniform area of earth-tone color (brown/tan) with little to no vegetation or man-made structures.",
    "construction_site": "An irregular area with disturbed soil, scattered equipment, and partially constructed geometric shapes.",
    "farmland": "A patchwork of regular geometric parcels with varying vegetation textures, often with circular center-pivot irrigation patterns.",
    "helipad": "A circular or H-shaped marking on a flat surface, typically on a building rooftop or open area.",
}

CONTOUR_DESCRIPTIONS = {
    "linear": "A long, straight or gently curved shape with high aspect ratio and parallel edges, like bridges or runways.",
    "compact": "A shape with low aspect ratio and well-defined closed boundary, like vehicles or storage tanks.",
    "elongated": "A narrow, extended shape with pointed or rounded ends, like ships or airplanes.",
    "complex": "A shape composed of multiple connected components or irregular boundaries, like harbors or service areas.",
    "large_rectangular": "A large area with four straight sides and right-angle corners, like buildings or sports fields.",
}

FIXED_BOUNDARY_CATEGORIES = [
    "airplane", "baseballfield", "bridge", "chimney",
    "Expressway-toll-station", "groundtrackfield", "ship",
    "storagetank", "tenniscourt", "vehicle", "windmill",
]

NONFIXED_BOUNDARY_CATEGORIES = [
    "airport", "basketballcourt", "dam", "Expressway-Service-area",
    "golffield", "harbor", "overpass", "stadium", "trainstation",
]
