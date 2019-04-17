# Parking Spot Detector

In this project, we developed a real-time parking spot detection system that uses a deep neural network. This model predicts and localizes vacant and occupied parking spots in videos and images captured from recording devices. To detect multiple parking spots in real time we used Single Shot Multibox Detection (SSD) with VGG16 as our base model. Furthermore, we explored rotated vs. non-rotated bounding boxes in terms of computational complexity and parking spot detection quality.


## Full Report:

<object data="ParkingSpotDetectionReport.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="ParkingSpotDetectionReport.pdf">
        <p>You can view the Report at: <a href="ParkingSpotDetectionReport.pdf">View Report</a>.</p>
    </embed>
</object>


## Results:

These two videos show preliminary results of our work:

[![Moving Camera](http://img.youtube.com/vi/exzxPoTZhBI/0.jpg)](http://www.youtube.com/watch?v=exzxPoTZhBI)

[![Steady Camera](http://img.youtube.com/vi/yiqaW-X2kuE/0.jpg)](http://www.youtube.com/watch?v=yiqaW-X2kuE)


### To run notebooks:

In order for the jupyter notebook files to work, you need to include unzip PKLOT dataset such that the images for each day and weather are in "PKLot/PKLot/\*/\*/\*/\* .jpg" format.


## Collaborators:

Raymond, Brennan, Krishen, Timothy, and Pooya.


## Acknowledgement:

I would like to acknowledge @sgrvinod for his amazing tutorial at: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection 

Other references are included in the paper.


