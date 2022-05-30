# HW2 

## Project structure:
📁 online_inference/  
├─📄 test_predict.py  &emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp;&nbsp;       <-- tests  
├─📄 Dockerfile  
├─📄 fast.py  &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp;  <-- app  
├─📁 models/  &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;  <-- trained models  
├─📁 test_data/  &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp;&nbsp; <-- data for inference   
├─📄 requester_script.py  &emsp;&emsp;&emsp;&emsp; <-- script that makes requests to the app  
├─📄 README.md  
├─📁 src/  &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp; <-- inference pipeline source code  
├─📄 requirements.txt  
└─📁 configs/  

### Run without docker: 
**Inside online_inference:**  
  
**1. run the app:**  
``python -m fast`` 

**from a new terminal:**  
**2. run requests:**  
``python -m requester_script``  

**3. or run tests:**  
``python -m pytest test_predict.py``  

### Local docker:  
**Inside online_inference:**  

**4. build a docker image:**  
``docker build -t online .``  

**5. run a docker image:**  
``docker run -p 8000:8000 online``  

**now steps 2 and 3 are available from a new terminal** 

### Dockerhub:
**6. download docker image:**  
``docker pull ankehooliganke/online``  

**7. run downloaded image**  
``docker run -p 8000:8000 ankehooliganke/online`` 

**now steps 2 and 3 are available from a new terminal** 
