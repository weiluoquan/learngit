import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 获取包目录
    package_dir = get_package_share_directory('teamX_challenge')
    
    # 参数文件路径
    param_file = os.path.join(package_dir, 'config', 'param.yaml')
    
    # 检查参数文件是否存在
    if not os.path.exists(param_file):
        print(f"Warning: Parameter file not found: {param_file}")
        print("Node will use default parameters declared in code.")
        param_file = ""  # 不传递参数文件，让节点使用代码中的默认值
    
    # 创建视觉节点
    vision_node = Node(
        package='teamX_challenge',
        executable='vision_node',  # 可执行文件名称
        name='vision_node',        # 节点名称
        output='screen',
        parameters=[param_file] if param_file else []
        # 移除了 remappings，直接使用 /camera/image_raw
    )
    
    return LaunchDescription([
        vision_node,
    ])