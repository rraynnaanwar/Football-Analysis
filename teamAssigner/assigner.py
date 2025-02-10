import cv2
from sklearn.cluster import KMeans
class TeamAssigner:
    def __init__(self):
        self.teamColors = {}
        self.playerTeamDict = {}

    def assignTeamColors(self, frame, playerDetections):
        playerColors = []
        for __, playerDetection in playerDetections.items():
            boundingBox = playerDetection["bounding box"]
            playerColor = self.getPlayerColor(frame, boundingBox)
            playerColors.append(playerColor)
        self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        self.kmeans.fit(playerColors)
        self.teamColors[1] = self.kmeans.cluster_centers_[0]
        self.teamColors[2] = self.kmeans.cluster_centers_[1]


    def getClusteringModel(self, image):
        image2D = image.reshape(-1,3)
        kmeans = KMeans(n_clusters=2, init="k-means++", random_state=0).fit(image2D)
        return kmeans


    def getPlayerColor(self, frame, boundingBox):
        image = frame[int(boundingBox[1]):int(boundingBox[3]), int(boundingBox[0]):int(boundingBox[2])]
        
        topHalf = image[0:int(image.shape[0]/2), :]
        image2D = topHalf.reshape(-1,3)
        kmeans = self.getClusteringModel(image2D)
        labels = kmeans.labels_
        clusteredImage = labels.reshape(topHalf.shape[0], topHalf.shape[1])
        cornerClusters = [clusteredImage[0,0], clusteredImage[0,-1], clusteredImage[-1,0], clusteredImage[-1,-1]]
        nonPlayerClusters = max(set(cornerClusters), key=cornerClusters.count)
        playerClusters = 1-nonPlayerClusters
        playerColor = kmeans.cluster_centers_[playerClusters]
        return playerColor

    def getPlayerTeams(self, frame, playerBoundingBox, playerID):
        if playerID in self.playerTeamDict:
            return self.playerTeamDict[playerID]

        playerColor = self.getPlayerColor(frame, playerBoundingBox)
        teamID =  self.kmeans.predict(playerColor.reshape(1,-1))[0]
        teamID +=1
        if playerID==103:
            teamID =2
        self.playerTeamDict[playerID] = teamID
        return teamID
    