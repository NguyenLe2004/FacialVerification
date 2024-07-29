import React, { useRef, useState } from 'react';
import Webcam from 'react-webcam';
import { Button, CloseButton } from 'react-bootstrap';
import "./WebcamContainer.css"
import axios from 'axios';
const WebcamComponent = ({setEncodededUserImg, setIsOpenWebcam}) => {
  const [webcamSize, setWebcamSize] = useState({
    height : window.innerHeight ,
    width : window.innerWidth * 0.7
  })
  const webcamRef = useRef(null);

  const capture = async () => {
    try {
      const imageSrc = await webcamRef.current.getScreenshot();
      const data ={
        image : imageSrc
      }

      const response = await axios.post('http://127.0.0.1:4000/api/get_user_embedding', data, {
        headers: {
          'Content-Type': 'application/json'
        }
      });
      console.log(response.data);
      setEncodededUserImg(response.data);
      setIsOpenWebcam(false);
    } catch (error) {
      console.error('Lỗi khi tải ảnh lên:', error);
    }
  };

  return (
    <div>
      <div className='webcam-container-signup' > 
        <CloseButton className='close-icon' onClick={() => setIsOpenWebcam(false)}/>
        <Webcam
          ref={webcamRef}
          audio={false}
          height={webcamSize.height}
          width={webcamSize.width}
          screenshotFormat="image/jpg"
        />
        <div className="inner-rect"></div>
        <Button className='take-pic-btn' style={{transform:"scaleX(-1)",position:"absolute"}} onClick={capture}>Chụp ảnh</Button>
      </div>
    </div>

  );
};

export default WebcamComponent;