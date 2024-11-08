## Instruction 

1) git clone ```https://github.com/DB11051998/waysa-assignment.git```
2) Download SAVED_MODEL_MULTICLASS.pt from the shared drive link ``` https://drive.google.com/file/d/190wR5UCClhTR5UA3yecJz98fyqjISpuE/view?usp=drive_link ``` and should placed in the project directory
3) build the docker using the following command ```docker build -t {image-name} -f Dockerfile .```
4) then do ```docker run --rm --gpus all --shm-size=20gb -p 7600:7600 {image-name:tag-name}```
5) using postman hit the endpoint ```http://localhost:7600/predict```with the given formatted payload
```
{
  "text": "Just added my #SXSW flights to @planely. Matching people on planes/airports. Also downloaded the @KLM iPhone app, nicely done."
}
```

6) the output would be in below format

```
{
  "emotion": "Positive emotion",
  "target": "iPad or iPhone App"
}
```


## Training and Validation

1) Have used the pretrained model of BERT from huggingface package, as the task required to return two outputs i.e emotion and towards which product it is inclined to, so for that a two linear layers were used , which uses hidden output of the BERT pretrained model.
2) train_val.py is the training code

## Notebook

Contains the EDA in it, the same information is present in this google doc https://docs.google.com/document/d/1bvwQmIZbCJu6ujJIGpAvOEYPW5F7ic32kGt3wc8BXcw/edit?tab=t.0


