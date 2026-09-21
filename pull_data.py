from roboflow import Roboflow

print("Connecting to Roboflow...")
rf = Roboflow(api_key="e5vgK9uZozTJhVSwioRP")
project = rf.workspace("anikets-workspace-mkmbv").project("rash-bot")
version = project.version(1)

print("Downloading New Drivable Area Dataset...")
dataset = version.download("yolov8")
print(f"Download complete! Dataset saved to: {dataset.location}")
