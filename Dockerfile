FROM python:3.11
  
ARG GRADIO_SERVER_NAME="0.0.0.0"  
ENV GRADIO_SERVER_NAME=${GRADIO_SERVER_NAME}  
  
ARG GRADIO_SERVER_PORT="80"  
ENV GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT}

EXPOSE 80
      
RUN apt-get update && \    
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \ 
    apt-get install -y ffmpeg && \
    apt-get clean && \    
    rm -rf /var/lib/apt/lists/* 
    
WORKDIR /app    

COPY . /app    
     
RUN rm -rf /tmp/*    

RUN pip install -e .

CMD ["python", "./demo/gradio_demo.py"]