需要安装以下依赖：
nvitop
loguru
redis

并在安装好 redis 后，修改文件中的 REDIS_PATH 变量即可使用：

```
@gpu_queue        # GPU排队注解
def workflow():   # 训练函数主入口，该函数会进入排队程序
  pass
```
