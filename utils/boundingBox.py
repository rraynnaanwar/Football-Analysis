def getCenterOfBox(boundingBox):
    x1, y1, x2, y2 = boundingBox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def getWidthOfBox(boundingBox):
    x1, y1, x2, y2 = boundingBox
    return int(x2 - x1)

def measureDistance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**.5

def measureXYDistance(p1,p2):
    return p1[0]-p2[0],p1[1]-p2[1]
