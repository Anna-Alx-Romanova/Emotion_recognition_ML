import os
from app import app
from PIL import Image
def sentiment_analysis(filename):
    if filename == "":
        result = "No image to analysis"
        return result
    else:
        loaded_image1=os.path.join(app.config["IMAGE_UPLOADS_GLOBAL"], filename)
        im = Image.open(loaded_image1)
        width, height = im.size
        result_analysis = f'{filename} \n width ={width} and height ={height}'
       
        return result_analysis