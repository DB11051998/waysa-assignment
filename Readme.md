## Instruction 

1) git clone ```https://github.com/DB11051998/waysa-assignment.git```
2) Download SAVED_MODEL_MULTICLASS.pt from the shared drive link ``` https://drive.google.com/file/d/190wR5UCClhTR5UA3yecJz98fyqjISpuE/view?usp=drive_link ```
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


