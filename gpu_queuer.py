from functools import wraps
import datetime
from nvitop import select_devices, Device, GpuProcess, NA
import time
import os
from loguru import logger
import json
from redis import Redis
import psutil
import subprocess

REDIS_PATH = ""

class RedisClient:
    def __init__(self):
        self.client = Redis(
            host="127.0.0.1",
            port=6379,
            decode_responses=True,
            encoding="UTF-8",
        )
        is_start = False
        try:
            response = self.client.ping()
            is_start = True
        except Exception as e:
            pass
        if not is_start:
            logger.warning("Redis服务器连接失败，尝试启动服务...")
            # 捕获命令输出
            subprocess.run(
                [
                    REDIS_PATH + "/bin/redis-server",
                    REDIS_PATH + "/bin/redis.conf",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            time.sleep(5)
            response = self.client.ping()
            if response is None:
                raise Exception("Redis服务器启动失败，请检查Redis是否正确安装！")

    def get_self_occupied_gpus(self, only_gpus=True):
        """
        获取自己已经占用的Gpu序号
        """
        self_occupied_gpus = self.client.hgetall("running_processes")
        if only_gpus:
            all_gpus = []
            for task in self_occupied_gpus.values():
                gpus = [
                    int(device) for device in str(json.loads(task)["cuda"]).split(",")
                ]
                all_gpus.extend(gpus)
            return list(set(all_gpus))
        return [json.loads(g) for g in self_occupied_gpus.values()]

    def join_wait_queue(self, id, n_gpus, memo):
        """
        加入等待队列
        """
        curr_time = datetime.datetime.now()
        creat_time = datetime.datetime.strftime(curr_time, "%Y-%m-%d %H:%M:%S")
        content = {
            "n_gpus": n_gpus,
            "create_time": creat_time,
            "update_time": creat_time,
            "system_pid": os.getpid(),
            "id": id,
            "task_desc": memo,
        }
        wait_num = len(self.client.lrange("wait_queue", 0, -1))
        self.client.rpush("wait_queue", json.dumps(content))
        if wait_num == 0:
            logger.info("正在排队中！ 目前排第一位！")
        else:
            logger.info(f"正在排队中！ 前方还有 {wait_num} 个任务！")
        return wait_num

    def is_my_turn(self, id):
        """
        排队这么长时间，是否轮到我了？
        """
        curr_task = json.loads(self.client.lrange("wait_queue", 0, -1)[0])
        return curr_task["id"] == id

    def update_queue(self, id):
        """
        更新等待队列
        """
        task = json.loads(self.client.lrange("wait_queue", 0, -1)[0])
        if task["id"] != id:
            # 登记异常信息
            logger.warning("当前训练任务并不排在队列第一位，请检查Redis数据正确性！")
        curr_time = datetime.datetime.now()
        update_time = datetime.datetime.strftime(curr_time, "%Y-%m-%d %H:%M:%S")
        task["update_time"] = update_time
        self.client.lset("wait_queue", 0, json.dumps(task))

    def pop_wait_queue(self, id):
        """
        弹出当前排位第一的训练任务
        """
        task = json.loads(self.client.lrange("wait_queue", 0, -1)[0])
        if task["id"] != id:
            # 登记异常信息
            logger.warning("当前训练任务并不排在队列第一位，请检查Redis数据正确性！")
        next_task = self.client.lpop("wait_queue")
        return next_task

    def register_process(self, id, cuda, n_gpus, memo):
        """
        将当前训练任务登记到进程信息中
        """
        curr_time = datetime.datetime.now()
        creat_time = datetime.datetime.strftime(curr_time, "%Y-%m-%d %H:%M:%S")

        content = {
            "cuda": cuda,
            "units_count": n_gpus,
            "create_time": creat_time,
            "update_time": creat_time,
            "system_pid": os.getpid(),
            "id": id,
            "task_desc": memo,
        }
        self.client.hset("running_processes", id, json.dumps(content))
        logger.info("成功登记进程使用信息到Redis服务器！")
        return id

    def deregister_process(self, id):
        """
        删除当前训练任务的信息
        """
        task = self.client.hget("running_processes", id)
        if task:
            self.client.hdel("running_processes", id)
            logger.info("成功删除Redis服务器上的进程使用信息！")
        else:
            logger.warning(
                "无法找到当前训练任务在Redis服务器上的进程使用信息！或许可以考虑检查一下Redis的数据 🤔"
            )


class GPUQueuer:
    def __init__(self, visible_cuda="-1", n_gpus=1, memo="no memo"):
        """初始化GPUQueuer类

        Args:
            visible_cuda (str, optional): 可见的 GPU 编号，多个 GPU 编号用逗号分隔，如 "0,1,2,3". Defaults to "-1".
            n_gpus (int, optional): 需要的 GPU 数量. Defaults to 1.
            memo (str, optional): 任务备注. Defaults to "no memo".
        """
        self.visible_cuda = visible_cuda
        if visible_cuda == "-1" and os.environ.get("CUDA_VISIBLE_DEVICES"):
            self.visible_cuda = str(os.environ.get("CUDA_VISIBLE_DEVICES"))
        self.n_gpus = n_gpus
        self.memo = memo
        self.redis_client = RedisClient()
        self.devices = Device.all()
        curr_time = datetime.datetime.now()
        creat_time = datetime.datetime.strftime(curr_time, "%Y-%m-%d %H:%M:%S")
        self.id = str(os.getpid()) + str(
            int(time.mktime(time.strptime(creat_time, "%Y-%m-%d %H:%M:%S")))
        )

    @logger.catch
    def start(self):
        # ---------------------------------------------------------------------------- #
        #                         获取当前符合条件的所有处理器
        # ---------------------------------------------------------------------------- #
        self.clear_zombie_processes()
        self_occupied_gpus = self.redis_client.get_self_occupied_gpus()
        devices = Device.all()
        if self.visible_cuda != "-1":
            devices = [
                Device(index=int(device_id))
                for device_id in self.visible_cuda.split(",")
            ]

        devices = [
            device for device in devices if device.index not in self_occupied_gpus
        ]

        if len(devices) >= self.n_gpus:
            cuda = select_devices(
                devices=devices,
                format="index",
                min_count=self.n_gpus,
            )
            cuda = [str(x) for x in cuda]
            cuda = ",".join(cuda)
            self.cuda = cuda
            self.redis_client.register_process(
                self.id, cuda, n_gpus=self.n_gpus, memo=self.memo
            )
            logger.info(f"获取到足够的卡，当前分配的卡为：{cuda}")
            return cuda
        else:
            # ---------------------------------------------------------------------------- #
            #                         如果需要排队就送入队列
            # ---------------------------------------------------------------------------- #
            wait_num = self.redis_client.join_wait_queue(
                self.id, n_gpus=self.n_gpus, memo=self.memo
            )

        # ---------------------------------------------------------------------------- #
        #                         排队模式，等待处理器
        # ---------------------------------------------------------------------------- #
        logger.warning("当前没有足够的计算资源，正在排队...")
        wait_count = 0
        while not self.redis_client.is_my_turn(
            self.id
        ) or not is_processing_units_ready(devices, self.n_gpus):
            self.clear_zombie_processes()
            time.sleep(15)
            curr_time = str(time.strftime("%m月%d日 %H:%M:%S", time.localtime()))
            if self.redis_client.is_my_turn(self.id):
                # 更新队列
                self.redis_client.update_queue(self.id)

            wait_num = len(self.redis_client.client.lrange("wait_queue", 0, -1)) - 1
            print(
                f"\r更新时间: {curr_time} | 该任务（PID：{os.getpid()}）需要 {self.n_gpus} 块卡，前面还有 {wait_num} 个排队任务，已刷新 {wait_count} 次",
                end="",
                flush=True,
            )

            self_occupied_gpus = self.redis_client.get_self_occupied_gpus()

            devices = Device.all()
            if self.visible_cuda != "-1":
                devices = [
                    Device(index=int(device_id))
                    for device_id in self.visible_cuda.split(",")
                ]
            devices = [
                device for device in devices if device.index not in self_occupied_gpus
            ]
            wait_count += 1

        # ---------------------------------------------------------------------------- #
        #                         更新可用处理器
        # ---------------------------------------------------------------------------- #
        cuda = select_devices(
            devices=devices,
            format="index",
            min_count=self.n_gpus,
        )
        cuda = [str(x) for x in cuda]
        cuda = ",".join(cuda)
        self.cuda = cuda
        logger.info(f"获取到足够的卡，当前分配的卡为：{cuda}")

        # ---------------------------------------------------------------------------- #
        #                         从队列中弹出并注册处理器和进程
        # ---------------------------------------------------------------------------- #
        self.redis_client.pop_wait_queue(self.id)
        self.redis_client.register_process(
            self.id, cuda, n_gpus=self.n_gpus, memo=self.memo
        )

        return cuda

    @logger.catch
    def close(self):
        # 释放资源
        self.redis_client.deregister_process(self.id)

    def clear_zombie_processes(self):
        """清理僵尸进程"""
        self_occupied_gpus = self.redis_client.get_self_occupied_gpus(only_gpus=False)
        queue = self.redis_client.client.lrange("wait_queue", 0, -1)
        for task in self_occupied_gpus:
            pid = int(task["system_pid"])
            if not psutil.pid_exists(pid):
                self.redis_client.client.hdel("running_processes", task["id"])
                logger.info(f"发现 GPU占用信息 中存在残余数据，已清除，进程为{pid}")
        for task_json in queue:
            task = json.loads(task_json)
            pid = int(task["system_pid"])
            if not psutil.pid_exists(pid):
                self.redis_client.client.lrem("wait_queue", 1, task_json)
                logger.info(f"发现 GPU排队队列 中存在残余数据，已清除，进程为{pid}")

        gpu = {}

        devices = Device.all()  # or `Device.all()` to use NVML ordinal instead
        for device in devices:
            processes = device.processes()

            gpu["index"] = device.physical_index
            gpu["GPU utilization"] = f"{device.gpu_utilization()}%"
            gpu["Total memory"] = f"{device.memory_total_human()}"
            gpu["Used memory"] = f"{device.memory_used_human()}"
            gpu["Free memory"] = f"{device.memory_free_human()}"

            keys = self.redis_client.client.keys()
            for key in keys:
                if "GPU info --> " + str(device.physical_index) in key:
                    self.redis_client.client.delete(key)

            gpu_name = (
                "GPU info --> "
                + str(device.physical_index)
                + f" utilization {device.gpu_utilization()}%  Free memory {device.memory_free_human()}"
            )
            self.redis_client.client.set(gpu_name, json.dumps(gpu))

            new_processes = []
            if len(processes) > 0:
                processes = GpuProcess.take_snapshots(processes.values(), failsafe=True)
                processes.sort(key=lambda process: (process.username, process.pid))
                new_processes = []
                for snapshot in processes:
                    process = {}
                    process["pid"] = snapshot.pid
                    process["username"] = snapshot.username
                    process["time"] = snapshot.running_time_human
                    process["gpu_memory"] = (
                        snapshot.gpu_memory_human
                        if snapshot.gpu_memory_human is not NA
                        else "WDDM:N/A"
                    )
                    process["gpu_memory_percent"] = f"{snapshot.gpu_memory_percent}%"
                    process["command"] = snapshot.command
                    new_processes.append(process)
            if len(new_processes) > 0:
                self.redis_client.client.set(
                    "GPU " + str(device.physical_index) + " processes",
                    json.dumps(new_processes),
                )
            else:
                self.redis_client.client.delete(
                    "GPU " + str(device.physical_index) + " processes"
                )


def is_processing_units_ready(devices, n_gpus):
    if len(devices) < n_gpus:
        # 没有符合条件的处理器
        return False
    else:
        return True


def gpu_queue(func):
    """
    装饰器：等待GPU资源并记录排队时间
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # 记录开始时间
        start_time = datetime.datetime.now()

        # GPU排队
        queuer = GPUQueuer()
        queuer.start()

        # 计算排队时间
        end_time = datetime.datetime.now()
        time_diff = end_time - start_time

        # 计算天、小时、分钟、秒
        days = time_diff.days
        seconds = time_diff.seconds
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60

        queue_time = f"{days}天 {hours}小时 {minutes}分钟 {seconds}秒"
        logger.info(f"排队时间：{queue_time}")

        # 调用原始函数
        return func(*args, **kwargs)

    return wrapper
