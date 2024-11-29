import os
import csv
import yaml
import yt_dlp as youtube_dl

def create_directories(data_dir, csv_file):
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        categories = set(row['Category'] for row in reader)
    
    for category in categories:
        path = os.path.join(data_dir, 'abnormal', category)
        os.makedirs(path, exist_ok=True)

def download_video(args):
    url, output_path = args

    ydl_opts = {
        'outtmpl': output_path,
    }
    
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
            return True
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False

def download_video(url, output_path):
    
    ydl_opts = {
        'outtmpl': output_path,
    }
    
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
            return True
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False

def process_csv(data_dir, csv_file):
    create_directories(data_dir, csv_file)
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Availability'] == 'Unavailable' or not row['URL']:
                continue
            
            source = row['Source']
            if source == 'WEI BD': source = 'WEI_BD'
            category = row['Category']
            filename = f"{source}_{row['Video']}.mp4"
            output_path = os.path.join(data_dir, 'abnormal', category, filename)
            
            if os.path.exists(output_path):
                print(f"Skipping {filename} - Already exists")
                continue
            
            print(f"Downloading {filename}")
            download_video(row['URL'], output_path)

if __name__ == "__main__":
    
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    config_fp = f'{proj_dir}/config.yaml'

    with open(config_fp, 'r') as f:
        config = yaml.safe_load(f)
    
    data_dir = config['data_dir']
    csv_fp = config['csv_fp']
    
    process_csv(data_dir, f'{data_dir}/{csv_fp}')