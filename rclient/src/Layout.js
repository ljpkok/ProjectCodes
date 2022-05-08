import React from "react";
import { Layout, Menu} from 'antd';
import { UserOutlined, LaptopOutlined, NotificationOutlined } from '@ant-design/icons';
import UploadM from './UploadM'


const { Header, Content, Sider } = Layout;
const items1 = ['Menu', 'Help'].map((key) => ({
    key,
    label: `${key}`,
}));
const items2 = [UserOutlined, LaptopOutlined, NotificationOutlined].map((icon, index) => {
    const key = ['Classification', 'Upload', 'History'][index];
    return {
        key: `${key}`,
        icon: React.createElement(icon),
        label: `${key}`,
        children: new Array(4).fill(null).map((_, j) => {
        }),
    };
});

function LayoutDemo () {
    return (
        <Layout>
            <Header className="header">
                <div className="logo" />
                <Menu theme="dark" mode="horizontal" defaultSelectedKeys={['2']} items={items1} />
            </Header>
            <Layout>
                <Sider width={200} className="site-layout-background">
                    <Menu
                        mode="inline"
                        defaultSelectedKeys={['1']}
                        defaultOpenKeys={['sub1']}
                        style={{
                            height: '100%',
                            borderRight: 0,
                        }}
                        items={items2}
                    />
                </Sider>
                <Layout
                    style={{
                        padding: '0 24px 24px',
                    }}
                >
                    <Content
                        className="site-layout-background"
                        style={{
                            padding: 24,
                            margin: 0,
                            minHeight: 280,
                        }}
                    >
                        <UploadM/>
                    </Content>
                </Layout>
            </Layout>
        </Layout>
    )
}
export default LayoutDemo;