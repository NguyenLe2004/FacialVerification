import React, { useRef, useState, useEffect } from 'react';
import Webcam from 'react-webcam';
import { CloseButton } from 'react-bootstrap';
import "./WebcamContainer.css"
import axios from 'axios';

const WebcamComponent = ({userEncode, setUserEncode, setIsSignIn}) => {
  const [webcamSize, setWebcamSize] = useState({
    height: window.innerHeight,
    width: window.innerWidth * 0.7
  });
  const webcamRef = useRef(null);

  const sendImageToAPI = async () => {
    try {
      const imgSrc = webcamRef.current.getScreenshot();
      const res = await axios.post('http://127.0.0.1:4000/api/verify', { 
        image: imgSrc,
        user_encode : userEncode
       });
       if (res.data["probability"] > 0.95) {
        setUserEncode([]); // end verify
        setIsSignIn(true)
       } 
    } catch (error) {
      console.error('Error sending image to API:', error);
    }
  };

  useEffect(() => {
    const interval = setInterval(sendImageToAPI, 1000);

    return () => clearInterval(interval);
  }, [webcamRef, sendImageToAPI]);

  return (
    <div>
      <div className='webcam-container'>
      <CloseButton className='close-icon' onClick={() => setUserEncode([])}/>
        <Webcam
          ref={webcamRef}
          audio={false}
          height={webcamSize.height}
          width={webcamSize.width}
          screenshotFormat="image/jpg"
        />
        <div className="inner-rect"></div>
      </div>
    </div>
  );
};

export default WebcamComponent;