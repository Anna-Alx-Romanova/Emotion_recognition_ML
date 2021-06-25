import re
from app import app
#from PIL import Image
from flask import render_template, request, redirect, jsonify, make_response, url_for
import os
from werkzeug.utils import secure_filename
from flask import send_from_directory, abort
from sentiment_analysis import sentiment_analysis



app.config["IMAGE_UPLOADS"] = "app/static/img/uploads"
app.config["IMAGE_UPLOADS1"] = "static/img/uploads/"
app.config["CSV_UPLOADS"] = "static/img/csv/"
app.config["IMAGE_UPLOADS_GLOBAL"] = "c:/Users/Anna/Project/Diplom_project_v1/Emotion_recognition_ML/app/static/img/uploads/"

app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["PNG", "JPG", "JPEG", "GIF"]
app.config['MAX_IMAGE_FILESIZE'] = 50 * 1024 * 1024




@app.route("/", methods=["GET", "POST"])
def index():
    print(f"Flask ENV is set to: {app.config['ENV']}")
    if request.method == "POST":

        if request.files:

            if "filesize" in request.cookies:

                if not allowed_image_filesize(request.cookies["filesize"]):
                    print("Filesize exceeded maximum limit")
                    return redirect(request.url)

                image = request.files["image"]

                if image.filename == "":
                    print("No filename")
                    return redirect(request.url)

                if allowed_image(image.filename):
                    filename = secure_filename(image.filename)
                    print(filename)
                    image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))

                    print("Image saved")
                    loaded_image=os.path.join(app.config["IMAGE_UPLOADS1"], filename)
                    print(loaded_image)
                    result= sentiment_analysis(filename)
                    return render_template("public/index.html", loaded_image=loaded_image, result=result)

                else:
                    print("That file extension is not allowed")
                    return redirect(request.url)

    return render_template("public/index.html")



@app.route("/about")
def about():
    performer_name1 = "Daria IVANCHENKO"
    performer_name2 = "Anna ROMANOVA"

    return render_template("public/about.html", performer_name1=performer_name1, performer_name2=performer_name2)



def allowed_image(filename):
    if not "." in filename:
        return False
    # Split the extension from the filename
    ext = filename.rsplit(".", 1)[1]

    # Check if the extension is in ALLOWED_IMAGE_EXTENSIONS
    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False


def allowed_image_filesize(filesize):

    if int(filesize) <= app.config["MAX_IMAGE_FILESIZE"]:
        return True
    else:
        return False
