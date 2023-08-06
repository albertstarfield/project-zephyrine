#!/bin/bash
set -e
target0_Link_src_origin="https://github.com/albertstarfield/alpaca-electron-zephyrine-ggmlv2v3-universal"
target1_Link_branched_repo="https://github.com/Willy030125/alpaca-electron-GGML-v2-v3"
target1_branch="ProjectZephyrineGGMLv2v3-Implementation"
target0_branch="Willy-Alpaca-electron-GGML-v2-v3"
rootworkdir=$(pwd)/repoworkspace
rm -rf "${rootworkdir}"
git config user.name "Automated Reposync Bot"
git config user.email "albertstarfield2001@gmail.com"
mkdir "${rootworkdir}"
cd "${rootworkdir}"
git clone ${target0_Link_src_origin} target0
git clone ${target1_Link_branched_repo} target1

#target 1 sync
echo "Sync target0 to target1"
cd "${rootworkdir}"
cd target1
git checkout -b ${target1_branch}
rm -rf *
cp -rv ../target0/* .
git add -v .
git commit -m "Automated Project Zephyrine Repo Sync"
echo "Use your Classic github access token not your password"
echo "https://github.com/settings/tokens"
git push -f --set-upstream origin ${target1_branch}
git checkout main

echo "Sync target1 to target0"   
#target 0 sync
cd "${rootworkdir}"
cd target0
git checkout -b ${target0_branch}
rm -rf *
cp -rv ../target1/* .
git add -v .
git commit -m "Automated Project Zephyrine Repo Sync"
echo "Use your Classic github access token not your password"
echo "https://github.com/settings/tokens"
git push -f --set-upstream origin ${target0_branch}   
git checkout main

cd "${rootworkdir}"

rm -rf "${rootworkdir}"
echo "Done"

