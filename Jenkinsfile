#!/usr/bin/env groovy
   
// Jenkinsfle for waveflow build
//
// Quang Truong - Created: 02/09/2018

@Library('wavecomp') _

def nodeLabel = 'sw'

node(selectBuildNode(nodeLabel)) {
    def currentws = ''
    stage('Build') {
        try {
            deleteDir()

            dir("waveflow") {
                checkout scm // Repo waveflow
                echo "Using  repo 'waveflow', branch '$BRANCH_NAME'."
                currentws = pwd()
                
                echo 'Set python virtual environment ...'
                sh "virtualenv -p python3 ${currentws}"
                
                // Fix virtualenv pip issue: https://stackoverflow.com/questions/7911003/cant-install-via-pip-with-virtualenv
                sh """#!/bin/bash
                    cd ${currentws}
                    source bin/activate
                    bin/python bin/pip install -r requirements.txt
                    source wf_env.sh ${currentws}
                    wfbuild r"""
            }

        } catch (e) {
            currentBuild.result = "FAILED"
            notifyBuild(currentBuild.result, 'waveflow build')
            throw e
        }
    }
    stage('WaveFlow Tests') {
        try {
            dir("${currentws}/test") {
                echo 'waveflow Testing ...'
                sh """#!/bin/bash
                    . ${currentws}/bin/activate
                    . ${currentws}/wf_env.sh ${currentws}
                    pytest -n 4"""
            }
        } catch (e) {
            currentBuild.result = "FAILED"
            notifyBuild(currentBuild.result, 'waveflow test')
            throw e
        }
    }
}
