---
title: MLflow tracking server on AWS
blog_type: ml_notes
excerpt: Setting up an MLflow tracking server on AWS.
layout: post_with_toc
last_modified_at: 2023-04-14
---

### Mlflow tracking server setup in AWS
1. Start a `t4g.nano` or `t3.nano` server in US East (Ohio) as it's a super cheap option.
2. Find your IP and ssh command by clicking on the instance name -> Connect -> SSH Client tabs.
   `ssh -i "visp-admin-aws-keypair.pem" ubuntu@ec2-3-19-53-234.us-east-2.compute.amazonaws.com`
3. Create a new s3 bucket `numerai-v1`.
4. Copy aws credentials file. This credential file will contain a subset of aws
   access keys required for the server.
   `scp -i "visp-admin-aws-keypair.pem" ~/.aws/credentials_numerai ubuntu@ec2-3-19-53-234.us-east-2.compute.amazonaws.com:~/credentials`
5. SSH into the ec2 m/c and install aws command
    ``` shell
    # move aws credentials appropriate place \
    mkdir ~/.aws && mv ~/credentials ~/.aws/credentials && sudo apt-get update \
    # install the aws cli \
    && sudo apt-get install unzip  \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && unzip awscliv2.zip && sudo ./aws/install \
    # install docker \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - \
    && sudo add-apt-repository -y "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable" \
    && sudo apt install -y docker-ce \
    # Spin up mlflow server from https://github.com/flmu/mlflow-tracking-server \
    sudo docker run \
        --rm \
        --name mlflow-tracking-server \
        -p 5000:5000 \
        -e PORT=5000 \
        -e FILE_DIR=/mlflow \
        -e AWS_BUCKET="numerai-v1" \
        -e AWS_ACCESS_KEY_ID=`aws configure get visp_within_aws.aws_access_key_id` \
        -e AWS_SECRET_ACCESS_KEY=`aws configure get visp_within_aws.aws_secret_access_key` \
        foxrider/mlflow-tracking-server:0.2.0
   ```
