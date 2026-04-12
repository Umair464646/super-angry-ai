import QtQuick
import QtQuick.Controls

Frame {
    id: card
    property alias title: header.text
    property bool compact: false

    background: Rectangle {
        radius: 12
        color: "#121A27"
        border.color: "#1B2A41"
    }

    padding: compact ? 10 : 14

    Column {
        anchors.fill: parent
        spacing: compact ? 6 : 10

        Label {
            id: header
            font.pixelSize: 14
            font.bold: true
            color: "#D7E4FF"
        }

        Item {
            width: parent.width
            height: 1
            Rectangle {
                anchors.fill: parent
                color: "#1A2638"
            }
        }
    }
}
