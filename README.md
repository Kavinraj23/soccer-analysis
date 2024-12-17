# **Soccer Analysis System**

Computer vision project that utilizes YOLO-based object detection, pixel segmentation, and camera movement analysis to track players, analyze gameplay, and calculate speed/distance during football matches.

## **Technologies Used**
- **YOLOv8**: For real-time object detection of players.
- **KMeans Clustering**: For pixel segmentation and assigning player colors.
- **Optical Flow**: To detect and analyze camera movements.
- **Perspective Transformation**: For measuring distances accurately on the pitch.
- **Python**: Core programming language.
- **OpenCV**: For video processing and computer vision operations.
- **PyTorch**: To run and fine-tune YOLO models.
- **NumPy**: Data manipulation.

## **Future Updates**

1. **Data Export**  
   - Enable exporting analytical results, such as player tracking data, speed, and distance metrics, to CSV or JSON formats for further processing and reporting.

2. **Advanced Tracking Features**  
   - Implement tracking for advanced gameplay events such as:  
     - **Passes**: Detect and analyze passes between players.  
     - **Ball Possession**: Identify which player is in control of the ball as well as correctly identifying goalkeepers' teams.
     - **Player Heatmaps**: Visualize player movements and positions throughout the match.

3. **Model Training on a Larger Dataset**  
   - Retrain the YOLO model on a significantly larger dataset (beyond the initial 600 images) to improve detection accuracy and generalization across diverse match scenarios.

4. **Web-Based Dashboard**  
   - Develop an interactive web-based dashboard to:  
     - Visualize player tracking data and metrics.  
     - Provide heatmaps, paths, and game analysis results.  
     - Allow users to upload videos and customize analysis settings.  

## **Credits**

This project is inspired by and builds upon the following resources:
- (https://github.com/abdullahtarek/football_analysis)  
- YouTube Tutorial: [Football Player Tracking Using YOLO](https://www.youtube.com/watch?v=neBZ6huolkg)  
