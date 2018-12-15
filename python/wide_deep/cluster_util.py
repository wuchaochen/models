import json
import os


def cluster_to_estimator(cluster_str):
    cluster = json.loads(cluster_str)
    worker_0 = cluster['worker'][0]
    del (cluster['worker'][0])
    cluster['chief'] = [worker_0]
    return cluster


def export_cluster_env(cluster_str, job_name, index):
    cluster = cluster_to_estimator(cluster_str)
    if 'ps' == job_name:
        task_type = 'ps'
        task_index = index
    elif 'worker' == job_name:
        if 0 == index:
            task_type = 'chief'
            task_index = 0
        else:
            task_type = 'worker'
            task_index = index - 1

    os.environ['TF_CONFIG'] = json.dumps(
        {'cluster': cluster,
         'task': {'type': task_type, 'index': task_index}})
    print (os.environ['TF_CONFIG'])
    return cluster


if __name__ == '__main__':
    cluster_str_test = '''{"ps":["172.17.0.1:42593"],"worker":["172.17.0.1:35120","172.17.0.1:51114"]}'''
    export_cluster_env(cluster_str_test, 'ps', 0)
