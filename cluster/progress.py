from ray.util.state import list_tasks

for task in list_tasks(address="http://127.0.0.1:8265", filters=[("state", "=", "RUNNING")]):
    print(task.name, "pid=", task.worker_pid, flush=True)
