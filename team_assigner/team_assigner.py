from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def clustering_model(slef, image_top_half):
        #convert image_top_half into 2D array
        image_2d = image_top_half.reshape(-1,3) # -1 to automatically get image dimention, 3 so that each array have 3 columns (R,G,B)

        # group the pixels into 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1) # "k-means++" is faster and more efficient than the default initialization
        kmeans.fit(image_2d)

        return kmeans
    
    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] # bbox[1] = y1, bbox[3] = y2, bbox[0] = x1, bbox[2] = x2
        image_top_half = image [0:int(image.shape[0]/2),:]

        # initializing clustering model
        kmeans = self.clustering_model(image_top_half)

        # get cluster labels for each pixel
        labels = kmeans.labels_

        # reshape the labels to image shape
        clustered_image = labels.reshape(image_top_half.shape[0], image_top_half.shape[1])

        # get player cluster
        corner_clusters = [clustered_image[0,0], clustered_image[0,-1], clustered_image[-1,0], clustered_image[-1,-1]] # list of corner clusters
        non_player_clusters = max(set(corner_clusters), key=corner_clusters.count) # most frequent clusters in corners are the non_player_clusters
        player_cluster = 1 - non_player_clusters

        # get player color
        player_color = kmeans.cluster_centers_[player_cluster] # gets the average color of player 

        return player_color
    
    def assign_team_color(self, frame, players_detections):

        players_colors = []
        for player_id, player_detection in players_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            players_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        kmeans.fit(players_colors)

        self.kmeans = kmeans # to call it later

        self.team_colors[0] = kmeans.cluster_centers_[0]
        self.team_colors[1] = kmeans.cluster_centers_[1]

    def assign_player_team(self, frame, bbox, player_id): # bbox belong to player
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, bbox)

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]      # team id will be 0 or 1

        self.player_team_dict[player_id] = team_id

        return team_id