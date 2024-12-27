#!/usr/bin/env bash

# Mật khẩu bạn muốn đặt
PASSWORD="mypassword"

# Yêu cầu người dùng nhập mật khẩu
read -s -p "Nhập mật khẩu để vào session: " input_password
echo

if [ "$input_password" == "$PASSWORD" ]; then
    # Nếu mật khẩu đúng, nối vào session
    tmux attach -t secure_session
else
    echo "Mật khẩu sai. Không thể vào session."
    exit 1
fi