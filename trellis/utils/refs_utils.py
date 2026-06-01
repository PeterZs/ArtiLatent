# reference of object categories
cat_ref = {
    "Table": 1,
    "Dishwasher": 2,
    "StorageFurniture": 3,
    "Refrigerator": 4,
    "WashingMachine": 5,
    "Microwave": 6,
    "Oven": 7,
    "Safe": 8,
}

# reference of semantic labels for each part
sem_ref = {
    "fwd": {
        "door": 1,
        "drawer": 2,
        "base": 3,
        "handle": 4,
        "wheel": 5,
        "knob": 6,
        "shelf": 7,
        "tray": 8
    },
    "bwd": {
        1: "door",
        2: "drawer",
        3: "base",
        4: "handle",
        5: "wheel",
        6: "knob",
        7: "shelf",
        8: "tray"
    }
}

# reference of joint types for each part
joint_ref = {
    "fwd": {
       "fixed": 1,
        "revolute": 2,
        "prismatic": 3,
        "screw": 4,
        "continuous": 5 
    },
    "bwd": {
        1: "fixed",
        2: "revolute",
        3: "prismatic",
        4: "screw",
        5: "continuous"
    } 
}


import plotly.express as px
# pallette for joint type color
joint_color_ref = px.colors.qualitative.Set1
# pallette for graph node color
graph_color_ref = px.colors.qualitative.Bold + px.colors.qualitative.Prism
# pallette for semantic label color
semantic_color_ref = px.colors.qualitative.Vivid_r