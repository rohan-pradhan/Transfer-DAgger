import train_policy
import train_policy_one_shot
import racer
import argparse
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--n_epochs", type=int, help="number of epochs", default=50)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=256)
    parser.add_argument("--n_steering_classes", type=int, help="number of steering classes", default=20)
    parser.add_argument("--train_dir", help="directory of training data", default='./dataset/train')
    parser.add_argument("--validation_dir", help="directory of validation data", default='./dataset/val')
    parser.add_argument("--weights_out_file", help="where to save the weights of the network e.g. ./weights/learner_0.weights", default='')
    parser.add_argument("--dagger_iterations", help="", default=1)
    parser.add_argument("--freeze_last_n_layers", help="", default=0)
    parser.add_argument("--transfer_learning", help="", default='')
    parser.add_argument("--train_layers", type=int,help="", default=0) #1 = train conv layers, 2 = train evaluation layers, 3 = train last layer
    parser.add_argument("--learner_weights", help="", default='')
    parser.add_argument("--road_color", help="", default='')
    args = parser.parse_args()


    if (args.transfer_learning == 'True'):
        args.weighted_loss = True
        args.transfer_learning=True
        args.weights_out_file = "./weights/learner_ONE_SHOT_TRANSFER_{:d}_rc_{:s}.weights".format(args.train_layers, args.road_color)
        cumulative_rewards = []

        args.save_expert_actions = True
        args.expert_drives = True
        args.run_id = 0
        args.timesteps= 100000
        os.mkdir("./dataset/train_one_shot_TRANSFER_{:d}_rc_{:s}".format(args.train_layers, args.road_color))
        args.out_dir = "./dataset/train_one_shot_TRANSFER_{:d}_rc_{:s}".format(args.train_layers, args.road_color)
        args.train_dir="./dataset/train_one_shot_TRANSFER_{:d}_rc_{:s}".format(args.train_layers, args.road_color)

        print ('GETTING ONE SHOT EXPERT DEMONSTRATIONS')
        racer.run(None,args)
        print ('RETRAINING LEARNER ON AGGREGATED DATASET')

        policy = train_policy_one_shot.main(args)

    else:
        args.weighted_loss = True
        args.transfer_learning = False
        args.weights_out_file = "./weights/learner_ONE_SHOT_FULL_{:d}_rc_{:s}.weights".format(args.train_layers, args.road_color)
        cumulative_rewards = []

        args.save_expert_actions = True
        args.expert_drives = True
        args.run_id = 0
        args.timesteps= 100000
        os.mkdir("./dataset/train_one_shot_FULL_rc_{:s}".format(args.road_color))
        args.out_dir = "./dataset/train_one_shot_FULL_rc_{:s}".format(args.road_color)
        args.train_dir="./dataset/train_one_shot_FULL_rc_{:s}".format(args.road_color)
        args.freeze_last_n_layers = 0
        print ('GETTING ONE SHOT DEMONSTRATIONS')
        racer.run(None,args)
        print ('RETRAINING LEARNER ON AGGREGATED DATASET')
        
        policy = train_policy_one_shot.main(args)
    args.expert_drives = False


    cumulative_rewards.append(racer.run(policy, args))
    print(cumulative_rewards)

    plt.plot(figsize=(8, 4))


    plt.plot(cumulative_rewards)
    plt.grid(True)
    plt.legend(loc='right')
    plt.title('Cumulative Reward histograms')
    plt.xlabel('Dagger iterations')
    plt.ylabel('Rewards')


    plt.show()


    #
