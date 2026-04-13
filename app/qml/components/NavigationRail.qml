import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    id: root
    width: 220
    color: "#0B1018"
    border.color: "#1A2638"
    property int currentIndex: 0
    signal indexChanged(int index)

    ListModel {
        id: navModel
        ListElement { label: "Home"; icon: "⌂" }
        ListElement { label: "Data"; icon: "◫" }
        ListElement { label: "Strategy"; icon: "⚙" }
        ListElement { label: "Evolution"; icon: "⇅" }
        ListElement { label: "Neural"; icon: "◉" }
        ListElement { label: "Backtest"; icon: "▦" }
        ListElement { label: "Results"; icon: "★" }
        ListElement { label: "Export"; icon: "⇩" }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 12
        spacing: 8

        Label {
            text: "Crypto Strategy Lab"
            color: "#E6F0FF"
            font.pixelSize: 17
            font.bold: true
            Layout.fillWidth: true
            wrapMode: Text.Wrap
        }

        Repeater {
            model: navModel
            delegate: Rectangle {
                required property int index
                required property string label
                required property string icon
                Layout.fillWidth: true
                height: 42
                radius: 10
                color: root.currentIndex === index ? "#143358" : (hovered ? "#121E30" : "transparent")
                border.color: root.currentIndex === index ? "#2CA4FF" : "#1A2638"
                property bool hovered: false

                Row {
                    anchors.fill: parent
                    anchors.leftMargin: 12
                    anchors.rightMargin: 12
                    spacing: 10
                    Label { text: icon; color: "#8EC8FF"; font.pixelSize: 14 }
                    Label { text: label; color: "#D8E9FF"; font.pixelSize: 13 }
                }

                MouseArea {
                    anchors.fill: parent
                    hoverEnabled: true
                    onEntered: parent.hovered = true
                    onExited: parent.hovered = false
                    onClicked: {
                        root.currentIndex = index
                        appState.logUiEvent("Navigation clicked: " + label)
                        root.indexChanged(index)
                    }
                }
            }
        }
        Item { Layout.fillHeight: true }
    }
}
