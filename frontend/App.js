import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, ActivityIndicator, ProgressBarAndroid } from 'react-native';
import { Video } from 'expo-av';
import * as ImagePicker from 'expo-image-picker';
import { initializeApp } from 'firebase/app';
import { getStorage, ref, uploadBytesResumable, getDownloadURL } from 'firebase/storage';

// Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyDQKPtaRvNxpBp-YWzgcVjxnNOVA_sEOwE",
  authDomain: "crema-takehome.firebaseapp.com",
  projectId: "crema-takehome",
  storageBucket: "crema-takehome.appspot.com",
  messagingSenderId: "561553746659",
  appId: "1:561553746659:web:4907152bdd320c1ae7f3d6"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const storage = getStorage(app);

export default function App() {
  const [video, setVideo] = useState(null);
  const [processedVideo, setProcessedVideo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(null);

  const pickVideo = async () => {
    try {
      const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (!permissionResult.granted) {
        alert("You need to allow access to your media library to upload videos!");
        return;
      }

      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Videos,
        allowsEditing: true,
        quality: 1,
      });

      if (!result.canceled) {
        setVideo(result.assets[0]);
        setProcessedVideo(null);
        setError(null);
        setProgress(null);
      }
    } catch (err) {
      console.error("Error picking video:", err);
      setError("Failed to pick video");
    }
  };

  const listenToProgress = (taskId) => {
    const eventSource = new EventSource(`http://localhost:5000/api/progress/${taskId}`);
    
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log('Progress update:', data);
      
      if (data.status === 'complete') {
        setProgress(null);
        setProcessedVideo({ uri: data.video_url });
        eventSource.close();
      } else if (data.status === 'error') {
        setError(data.message);
        setProgress(null);
        eventSource.close();
      } else {
        setProgress({
          status: data.status,
          value: data.progress / 100
        });
      }
    };

    eventSource.onerror = (error) => {
      console.error('SSE Error:', error);
      setError('Lost connection to server');
      setProgress(null);
      eventSource.close();
    };
  };

  const uploadAndProcessVideo = async () => {
    if (!video) return;

    setLoading(true);
    setError(null);
    setProgress(null);

    try {
      // First, upload to Firebase Storage
      const response = await fetch(video.uri);
      const blob = await response.blob();
      
      const filename = `uploads/${Date.now()}.mp4`;
      const storageRef = ref(storage, filename);
      
      // Upload the video
      const uploadTask = uploadBytesResumable(storageRef, blob);
      
      // Wait for upload to complete
      await new Promise((resolve, reject) => {
        uploadTask.on(
          'state_changed',
          (snapshot) => {
            const progress = (snapshot.bytesTransferred / snapshot.totalBytes);
            setProgress({
              status: 'uploading',
              value: progress
            });
          },
          reject,
          resolve
        );
      });

      // Get the download URL
      const downloadURL = await getDownloadURL(storageRef);

      // Start processing
      const processResponse = await fetch('http://localhost:5000/api/process-video', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          video_url: downloadURL,
        }),
      });

      const data = await processResponse.json();
      
      if (!processResponse.ok) {
        throw new Error(data.error || 'Failed to process video');
      }

      // Listen for progress updates
      listenToProgress(data.task_id);
      
    } catch (err) {
      console.error("Error processing video:", err);
      setError(err.message || "Failed to process video");
      setProgress(null);
    } finally {
      setLoading(false);
    }
  };

  const renderProgress = () => {
    if (!progress) return null;

    return (
      <View style={styles.progressContainer}>
        <Text style={styles.progressText}>
          {progress.status.charAt(0).toUpperCase() + progress.status.slice(1)}...
        </Text>
        <ProgressBarAndroid
          styleAttr="Horizontal"
          indeterminate={progress.value === undefined}
          progress={progress.value || 0}
          style={styles.progressBar}
        />
        {progress.value !== undefined && (
          <Text style={styles.progressText}>{Math.round(progress.value * 100)}%</Text>
        )}
      </View>
    );
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>TikTok Watermark Remover</Text>
      
      <TouchableOpacity style={styles.button} onPress={pickVideo}>
        <Text style={styles.buttonText}>Pick Video</Text>
      </TouchableOpacity>

      {video && (
        <View style={styles.videoContainer}>
          <Text style={styles.subtitle}>Original Video:</Text>
          <Video
            source={{ uri: video.uri }}
            style={styles.video}
            useNativeControls
            resizeMode="contain"
          />
        </View>
      )}

      {video && !loading && !processedVideo && !progress && (
        <TouchableOpacity 
          style={[styles.button, styles.processButton]} 
          onPress={uploadAndProcessVideo}
        >
          <Text style={styles.buttonText}>Process Video</Text>
        </TouchableOpacity>
      )}

      {renderProgress()}

      {error && (
        <Text style={styles.errorText}>{error}</Text>
      )}

      {processedVideo && (
        <View style={styles.videoContainer}>
          <Text style={styles.subtitle}>Processed Video:</Text>
          <Video
            source={{ uri: processedVideo.uri }}
            style={styles.video}
            useNativeControls
            resizeMode="contain"
          />
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    paddingTop: 50,
    paddingHorizontal: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  subtitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 10,
  },
  button: {
    backgroundColor: '#2196F3',
    padding: 15,
    borderRadius: 8,
    marginVertical: 10,
    width: '80%',
    alignItems: 'center',
  },
  processButton: {
    backgroundColor: '#4CAF50',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  videoContainer: {
    width: '100%',
    marginVertical: 15,
    alignItems: 'center',
  },
  video: {
    width: '100%',
    height: 300,
  },
  progressContainer: {
    width: '80%',
    marginVertical: 20,
    alignItems: 'center',
  },
  progressBar: {
    width: '100%',
    height: 20,
  },
  progressText: {
    marginVertical: 5,
    fontSize: 16,
    color: '#666',
  },
  errorText: {
    color: 'red',
    marginVertical: 10,
    textAlign: 'center',
  },
}); 