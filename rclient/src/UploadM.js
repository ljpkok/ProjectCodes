import React, {Component} from 'react';
import './App.css';
import {Button, message, Table, Upload} from 'antd';
import axios from 'axios';
import {UploadOutlined} from "@ant-design/icons";
import { Spin, Descriptions, Row , Col} from 'antd';

class UploadM extends Component {
  constructor(props){
    super(props)
    this.state={
      fileName: false,
      fileList:[],
      source:null,
      result: {class: null, obj: null},
    }
    this.file=null;
  }

  handleChange = (info)=> {
    const file = info.file.originFileObj
    console.log( file )
    const status = info.file.status;
    if (status !== 'uploading') {
      console.log(info.file, info.fileList);
    }
    if (status === 'done') {
      message.success(`${info.file.name} file uploaded successfully.`);
    } else if (status === 'error') {
      message.error(`${info.file.name} file upload failed.`);
    }
  }
  //手动上传
  handleUpload= () => {
    const { fileList } = this.state;
    const formData = new FormData();
    fileList.forEach((file) => {
      console.log(file)
      formData.append('files', file,file.name);
      this.setState({
        fileName: file.name,
        source:URL.createObjectURL(file)
      })
    });

    const config ={
      headers:{
        'Content-Type': 'multipart/form-data'
      }
    }
    axios.post('http://localhost:5000/upload', formData, config).then( res => {
      console.log(res)
      this.setState({
        result: {class : res.data.class, obj: res.data.obj}
      })
    }).catch( err => console.log(err))
  }

  render() {
    const uploadProps={
      action:'http://localhost:5000/upload',
      onRemove: (file) => {
        this.setState(({ fileList }) => {
          const index = fileList.indexOf(file);
          const newFileList = fileList.slice();
          newFileList.splice(index, 1);
          return {
            fileList: newFileList,
          };
        });
      },
      beforeUpload: (file) => {
        console.log('beforeUpload:',file)
        this.setState((state) => ({
          fileList: [...state.fileList, file],
        }));
        return false;
      },
      fileList: this.state.fileList,
    }
    return (
      <div className="App">
        <Row gutter={[16, 32]}>
          <h1>Welcome to video classification system: {'v1.0.0'}</h1>
        </Row>
        <Row gutter={[16, 16]}>
          <Col>
            <Upload {...uploadProps}>
            <Button>
              <UploadOutlined/> Select File
            </Button>
          </Upload>
          </Col>
          <Col>
            <Button
                className="upload-demo-start"
                type="primary"
                onClick={this.handleUpload}
                disabled={this.state.fileList.length === 0}
            >
              Start Upload
            </Button>
          </Col>
        </Row>

        <Row gutter={[16, 32]}>
          {this.state.source ? <video
              className="VideoInput_video"
              width="480"
              height= "320"
              controls
              src={this.state.source}
          /> : null}
        </Row>

        <Row gutter={[16, 32]}>

        </Row>
        <Row gutter={[16, 32]}>

          {this.state.result.class || !this.state.source?
              <Descriptions
                  title="Results"
                  bordered
                  column={{ xxl: 4, xl: 3, lg: 3, md: 3, sm: 2, xs: 1 }}
              >
                <Descriptions.Item label="Video Name">{this.state.fileName? this.state.fileList[0].name  : "None"}</Descriptions.Item>
                <Descriptions.Item label="Class">    {this.state.result.class ? this.state.result.class  : "None"}</Descriptions.Item>
                <Descriptions.Item label="Objects">
                  {this.state.result.obj ?
                      this.state.result.obj
                      : "None"}
                </Descriptions.Item>
              </Descriptions> : <Spin />}
        </Row>

      </div>
    );
  }
}

export default UploadM;
