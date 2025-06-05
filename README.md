Install Python 3.13.3
Install anaconda
## สร้าง Env ด้วย Conda
·         conda create --name ProjectLPR python=3.10
conda create --name ProjectLPR_Train python=3.10
·         conda activate ProjectLPR
·         conda deactivate
·         ^Z
·         pip list
·         conda install -c conda-forge ultralytics=8.3.3
·         conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=12.1 ultralytics=8.3.3
·         pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
·         conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
·         pip install opencv-contrib-python
·         pip install shapely
4.  	Install libraries
·         pip install flask
·         pip install opencv-python-headless
·         pip install ultralytics
·         pip install numpy
https://github.com/ultralytics/ultralytics

 
5.  	Computer specification      https://www.proxpc.com/blogs/system-hardware-requirements-for-yolo-in-2025?utm_source=chatgpt.com

## Database Integration

The application now uses a SQLite database (`vehicle.db`) to store registered
vehicles and detection logs.  Routes were added to manage this data:

- `POST /register_vehicle` – register a new plate in the database. JSON body
  should contain `plate` and optional `province`.
- `GET /vehicles` – list all registered vehicles.
- `GET /detections` – retrieve detection history from the database.

Detected plates are automatically checked against registered records and
logged with the result.
