import sys
import os
import argparse

# Thêm thư mục src vào sys.path để các module bên trong import lẫn nhau mà không cần sửa code gốc
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(src_path)

def main():
    parser = argparse.ArgumentParser(description="Run Anti-Jamming DRL Algorithms")
    parser.add_argument(
        "--mode", 
        type=str, 
        default="dqn", 
        choices=["q_learning", "dqn"],
        help="Select algorithm to run: q_learning or dqn (Default: dqn)"
    )
    
    args = parser.parse_args()

    if args.mode == "q_learning":
        print("====== STARTING Q-LEARNING AGENT ======")
        import q_learnning
        agent = q_learnning.q_learning_agent()
        agent.learning()
    elif args.mode == "dqn":
        print("====== STARTING DEEP Q-LEARNING AGENT ======")
        import deep_q_learning
        agent = deep_q_learning.deep_q_learning_agent(dueling=True)
        agent.learning()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(0)
