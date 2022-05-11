# Mini Project

1. Take picture using cameraCalib.py
2. Find homography matrix using Detect_lego_and_calibrate points.ipynb
3. VGIS_RV-miniproject.rdk is ready to run
4. uncomment line 12 in create_figures.py within the robodk environment and change line 235 to robot ip
# Image processing steps
## Initial image
![CapturedImage](https://user-images.githubusercontent.com/45823340/167778541-cf4b2391-7d58-4c3b-b12b-fb68d525dc25.png)

## Gamma mapping
![1_gamma_mapped](https://user-images.githubusercontent.com/45823340/167778552-028b8fa1-a244-41db-a1d5-2d6cca51fded.png)

## blurring
![2_blurred](https://user-images.githubusercontent.com/45823340/167778575-ea3217f4-5082-4e28-b461-bcdd970d74ae.png)

## K-means clustering
![3_kmean](https://user-images.githubusercontent.com/45823340/167778596-42892e0c-b97c-43b8-91a1-18b65c7ee19d.png)


## Colorspace conversion and threshold
![4_colorspace_conv_and_thresh](https://user-images.githubusercontent.com/45823340/167808436-e0196516-f57b-4539-9e85-fb2edea0573a.png)


## End result
![5_detected](https://user-images.githubusercontent.com/45823340/167778628-fb9892af-eb2b-4fb8-a622-84df85d01100.png)
