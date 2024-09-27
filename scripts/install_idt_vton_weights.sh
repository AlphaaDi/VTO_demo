apt update
apt-get install git-lfs rsync
git lfs install
cd ../..
mkdir vton_origin
git clone https://huggingface.co/spaces/yisol/IDM-VTON vton_origin
rm -rf vton_origin/.git/
rm -rf vton_origin/example/
rsync -av --ignore-existing vton_origin/ VTO_demo/