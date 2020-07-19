# Malaria Api
## This Malaria Api is for this  [Paper]() 
*For more information on the model and how we constructed the Deeplearning Model Refer the paper*

# Development of RestApi
For development of the RestApi we have used [Flask](https://flask.palletsprojects.com/en/1.1.x/) is a micro web framework written in Python.
It's the lightweight framework used to develop RestApi using Python.We choose Flask because it best suits our needs.
We embededded our model into the api.We have created two Routes one is default "/" and second is /api/predict.
We then Hosted our api on [Heroku](https://www.heroku.com/).

> * First Route = "https://malareaapi333.herokuapp.com/" -- [default](https://malareaapi333.herokuapp.com/)
if we visite the route it will display "Api is working go to /api/predict to get predction with img as input".
> * Second Route = "https://malareaapi333.herokuapp.com/api/predict" -- predicton route this route accepts base64 encoded image and return "Parasitized","Uninfected" class for the input Image

# Json Endpoints
Routes | Description | URL
------------ | ------------- | -------------
"/" | default route used for checking working of api | <https://malareaapi333.herokuapp.com/>
"/api/predict" | This route Accepts base64 encoded image and return "Parasitized","Uninfected" class based for input image | <https://malareaapi333.herokuapp.com/api/predict>

# Example for "/api/predict" Route
```javascript
{
 message: 
  { 
    image: base64Image 
  }
}
```
it accepts json object as input as shown in the above example.
>  Note : *The json object  as above mentioned did not change message and image name just supply base64 encoded image in place of base64Imag*

