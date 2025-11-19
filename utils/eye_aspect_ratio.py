from math import hypot

# EAR formula: (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
# Points are expected as 6 (x,y) tuples in order [p1..p6]

def eye_aspect_ratio(eye_points):
    if len(eye_points) != 6:
        raise ValueError("eye_points must be a list of 6 (x,y) tuples")
    p1, p2, p3, p4, p5, p6 = eye_points
    A = hypot(p2[0] - p6[0], p2[1] - p6[1])
    B = hypot(p3[0] - p5[0], p3[1] - p5[1])
    C = hypot(p1[0] - p4[0], p1[1] - p4[1])
    if C == 0:
        return 0.0
    ear = (A + B) / (2.0 * C)
    return ear
