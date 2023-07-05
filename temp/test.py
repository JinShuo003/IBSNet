def random_points():
    return [[]]


def IBS_Net(points):
    return [[]]


threshold_target = 0.001
threshold = 0.5
points = random_points()
udf1, udf2 = IBS_Net(points)

while threshold > threshold_target:
    seeds = []
    for i, point in enumerate(points):
        if abs(udf1[i] - udf2[i]) < threshold:
            seeds.append(point)
    if threshold > threshold_target:
        threshold /= 2
