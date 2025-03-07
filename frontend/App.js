import React, { useState, useEffect } from 'react';
import { StyleSheet, View, Text, TouchableOpacity, ActivityIndicator, SafeAreaView } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Video } from 'expo-av';

export default function App() {
  const [video, setVideo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Add error boundary effect
  useEffect(() => {
    // Clear any previous errors
    setError(null);
  }, []);

  const pickVideo = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Request permission
      const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
      
      if (!permissionResult.granted) {
        setError('Permission to access media library is required!');
        return;
      }
      
      // Launch image picker with video option - fixed mediaTypes value
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ['videos'],
        allowsEditing: true,
        quality: 1,
      });

      if (result.canceled) {
        console.log('User cancelled video picker');
      } else {
        // Video selected successfully
        const selectedVideo = result.assets[0];
        console.log('Video selected:', selectedVideo);
        setVideo(selectedVideo);
      }
    } catch (error) {
      console.error('Error picking video:', error);
      setError('Failed to pick video: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        <Text style={styles.title}>Video Picker App</Text>
        
        {error && (
          <View style={styles.errorContainer}>
            <Text style={styles.errorText}>{error}</Text>
          </View>
        )}
        
        <TouchableOpacity 
          style={styles.button} 
          onPress={pickVideo}
          disabled={loading}
        >
          <Text style={styles.buttonText}>
            {loading ? 'Loading...' : 'Pick a Video'}
          </Text>
        </TouchableOpacity>
        
        {loading && <ActivityIndicator size="large" color="#0000ff" />}
        
        {video && (
          <View style={styles.videoContainer}>
            <Text style={styles.videoInfo}>
              Selected Video
            </Text>
            {video.fileSize && (
              <Text style={styles.videoInfo}>
                Size: {(video.fileSize / 1024 / 1024).toFixed(2)} MB
              </Text>
            )}
            {video.duration && (
              <Text style={styles.videoInfo}>
                Duration: {(video.duration / 1000).toFixed(1)} seconds
              </Text>
            )}
            <Video
              source={{ uri: video.uri }}
              style={styles.videoPreview}
              useNativeControls
              resizeMode="contain"
              onError={(error) => {
                console.error('Video playback error:', error);
                setError('Failed to play video: ' + error);
              }}
            />
          </View>
        )}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  content: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 30,
  },
  button: {
    backgroundColor: '#2196F3',
    padding: 15,
    borderRadius: 5,
    marginBottom: 20,
    minWidth: 200,
    alignItems: 'center',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  videoContainer: {
    width: '100%',
    alignItems: 'center',
    marginTop: 20,
  },
  videoInfo: {
    fontSize: 16,
    marginBottom: 10,
  },
  videoPreview: {
    width: 300,
    height: 200,
    marginTop: 10,
    backgroundColor: '#f0f0f0',
  },
  errorContainer: {
    backgroundColor: '#ffebee',
    padding: 10,
    borderRadius: 5,
    marginBottom: 20,
    width: '100%',
  },
  errorText: {
    color: '#d32f2f',
    textAlign: 'center',
  }
}); 