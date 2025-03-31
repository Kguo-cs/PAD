import subprocess
import time
import sys

def cleanup_gpu_process(gpu_id):
    """Run shell commands to clean up GPU-related processes."""
    try:
        print(f"Cleaning up processes for GPU_ID: {gpu_id}...")
        # Shell commands for cleanup
        subprocess.run(f"ps -ef | grep 'graphicsadapter={gpu_id}' | awk '{{print $2}}' | xargs kill > /dev/null 2>&1 &", shell=True)
        subprocess.run(f"ps -ef | grep 'gpu-rank={gpu_id}' | awk '{{print $2}}' | xargs kill > /dev/null 2>&1 &", shell=True)
        subprocess.run(f"ps -ef | grep 'Carla' | awk '{{print $2}}' | xargs kill > /dev/null 2>&1 &", shell=True)
    except Exception as e:
        print(f"Error during cleanup: {e}")

def run_script_with_timeout(script, timeout=60):
    # Start the process
    process = subprocess.Popen(
        ['python', script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    last_output_time = time.time()

    try:
        # Continuously read output from stdout and stderr
        while True:
            output = process.stdout.readline()
            if output:
                sys.stdout.write(output)  # Print output to terminal immediately
                last_output_time = time.time()  # Reset the timeout timer when there's output
            
            # Check for any error output
            error = process.stderr.readline()
            if error:
                sys.stderr.write(error)  # Print error output to terminal immediately
                last_output_time = time.time()  # Reset the timeout timer when there's error output
            
            # Check if the process has finished
            if output == '' and error == '' and process.poll() is not None:
                break
            
            # Check if 60 seconds have passed with no new output
            if time.time() - last_output_time > timeout:
                print(f"Timeout: No output received for {timeout} seconds.")
                process.terminate()  # Terminate the process after timeout
                break

            time.sleep(1)  # Sleep for a short time to avoid busy-waiting

    except Exception as e:
        print(f"An error occurred: {e}")

    # Wait for the process to terminate properly
    process.wait()


while True:
    try:
        # Replace 'your_script.py' with the path to your Python script
        print("Starting script...")
        #result = run_script_with_timeout('leaderboard/leaderboard/pad_eval.py') #
        result=subprocess.run(['python', 'leaderboard/leaderboard/pad_eval.py'], check=True, timeout=3600)  # Run the script
        # If the script finishes correctly, break the loop (exit)
        print("Script finished successfully!")
        break
    except subprocess.CalledProcessError as e:
        # Handle error if the script terminates incorrectly
        print(f"Script terminated with error: {e}")
        cleanup_gpu_process(0)
        print("Restarting script...")
        time.sleep(1)  # Optional delay before restarting
    except Exception as e:
        # Handle any other unexpected exceptions
        print(f"Unexpected error: {e}")
        cleanup_gpu_process(0)
        print("Restarting script...")
        time.sleep(1)  # Optional delay before restarting
