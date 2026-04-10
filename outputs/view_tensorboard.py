import os
import subprocess
import sys

def main():
    """
    Launches TensorBoard to view logs stored in outputs/logs/
    """

    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "logs"))
    
    if not os.path.exists(log_dir):
        print(f"[ERROR] Log directory does not exist: {log_dir}")
        print("Make sure you have run the training pipeline first to generate logs.")
        sys.exit(1)

    print(f"[INFO] Starting TensorBoard for logs in: {log_dir}")
    print("[INFO] You can view the dashboard by opening the URL provided below in your browser.")
    print("[INFO] Press Ctrl+C in this terminal to stop the TensorBoard server.\n")
    
    try:
        # Launch TensorBoard
        subprocess.run([sys.executable, "-m", "tensorboard.main", "--logdir", log_dir])
    except KeyboardInterrupt:
        print("\n[INFO] TensorBoard server stopped by user.")
    except Exception as e:
        print(f"\n[ERROR] Failed to start TensorBoard: {e}")
        print("Ensure TensorBoard is installed: pip install tensorboard")

if __name__ == "__main__":
    main()
