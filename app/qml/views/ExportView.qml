import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    Layout.fillWidth: true
    Layout.fillHeight: true

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 12
        spacing: 10

        Label { text: "Export Center"; color: "#E2EEFF"; font.pixelSize: 20; font.bold: true }

        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            radius: 12
            color: "#0F1725"
            border.color: "#1B2A41"

            TextArea {
                anchors.fill: parent
                anchors.margins: 8
                text: appState.selectedStrategyJson()
                readOnly: true
                wrapMode: Text.Wrap
                color: "#D7E5FA"
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Button { text: "Copy Selected Strategy"; onClicked: appState.copySelectedStrategy() }
        }
    }
}
