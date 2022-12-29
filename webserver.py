from flask import Flask, render_template, request, redirect, url_for
from flask_ngrok import run_with_ngrok
from werkzeug.utils import secure_filename
from PIL import Image
import os

app = Flask(__name__)
run_with_ngrok(app)

data_list = []

@app.route('/', methods=['GET', 'POST'])
def main():
    return render_template('main.html')
 
@app.route('/fileUpload', methods = ['GET', 'POST'])
def file_upload():
    if request.method == 'POST':
        f = request.files['file']
        f_src = 'static/origin_web.jpg'
        
        f.save(f_src)
        return render_template('fileUpload.html')

@app.route('/fileUpload_cloth', methods = ['GET', 'POST'])
def fileUpload_cloth():
    if request.method == 'POST':
        f = request.files['file']
        f_src = 'static/cloth_web.jpg'
        
        f.save(f_src)
        return render_template('fileUpload_cloth.html')
 
@app.route('/view', methods = ['GET', 'POST'])
def view():
    print("inference start")
    
    terminnal_command ="python main.py"
    os.system(terminnal_command)
    
    print("inference end")
    return render_template('view.html', data_list=data_list)  # html을 렌더하며 DB에서 받아온 값들을 넘김
 
if __name__ == '__main__':
    app.run()