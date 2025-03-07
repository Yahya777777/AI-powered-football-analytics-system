def get_bbox_center(bbox):
    x1,y1,x2,y2 = bbox   #(x1,y1) is top left vertice, (x2,y2) down right vertice in bbox
    return int((x1+x2)/2), int((y1+y2)/2)

def get_bbox_width(bbox):
    return bbox[2] - bbox[0]  # returns x2 - x1