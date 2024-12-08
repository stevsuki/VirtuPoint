VirtuPoint is a virtual mouse application created using a Custom CNN Model and Hand Landmark Detection Technique

## Manual Book
The followings are the instructions on how to use VirtuPoint application

[VirtuPoint Manual Book.pdf](https://github.com/stevsuki/VirtuPoint/raw/virtupoint/Manual%20Book%20VirtuPoint.pdf)

## Installation
There are 2 ways to run this application, namely by downloading the .exe file on the gdrive link or running application independently by cloning the repository

### Run VirtuPoint.exe
Download VirtuPoint.exe from link drive:

https://drive.google.com/drive/folders/1fALJLQEbMm1Bdw2yV2-0G54K46VxTw8n?usp=sharing

### Run apps from Visual Studio Code:
1. Clone this repository
   ```
      git clone https://github.com/stevsuki/VirtuPoint.git
   ```
2. Install library dependencies:
   ```
      pip install -r requirements.txt
   ```
3. Run VirtuPoint.py
   ```
      python VirtuPoint.py
   ```
   if an error appears as below
   
   ![image](https://github.com/user-attachments/assets/c5bcbf61-01de-41d7-840d-ce6793591bb6)

   Please maksure Microsoft Visual C++ Redistributable Package has been installed

   if packaged not installed install msvc-runtime:
   ```
   pip install msvc-runtime==14.42.34433
   ```

