import cv2
import depthai as dai


def main():
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.Camera).build()
    preview = cam.requestOutput(
        (640, 480),
        type=dai.ImgFrame.Type.RGB888p
    )

    q = preview.createOutputQueue()

    pipeline.start()

    with pipeline:
        while pipeline.isRunning():
            frame = q.get().getCvFrame()
            cv2.imshow("OAK-D Test", frame)

            key = cv2.waitKey(1) & 0xFF
	    if key == ord('q'):
		break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
