## Flask Api Server Prac

### ec-2(ubuntu20.04)에 가상 환경을 구축하여 실행. -> 참고: https://flask.palletsprojects.com/en/2.2.x/installation/#install-flask

## setup & run

### virtual setup
<pre>
cd flask 
source venv/bin/activate 
</pre>

### run 
<pre>
 export FLASK_APP=app.py 
 flask run --host=0.0.0.0 # 클라우드(ec2)에서 실행시키기 때문에, public 접근 허용.
</pre>

## kill virtual
<pre>
deactivate
</pre>



